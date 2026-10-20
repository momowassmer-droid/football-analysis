"""Rebuild player tracking from the x-model auto-labels, with dedup.

Chain per clip (all outputs under data/output/<clip>/detections/):

    labels/*.txt --stage-1 dedup--> boxes --greedy track--> tracks
        --stage-2 co-travel dedup (appearance veto)--> x_raw.csv
        --pitch filter (cached camera motion)--> x_pitch.csv
        --enhanced stitch + smooth--> x_tracks.csv
        --team classification--> x_teams.csv

Every removed box is logged to x_dedup_log.csv (rule, removed box, kept
partner box) so the removals can be rendered and checked visually --
proving no real player lost coverage.

Usage:
    python -m tools.xmodel_pipeline            # all three test clips
    python -m tools.xmodel_pipeline match_left # one clip
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.analytics.team_classifier import classify_teams
from src.calibration.homography import PitchCalibration, filter_to_pitch_csv
from src.config.config_loader import load_config
from src.detection.dedup import Box, dedup_frame_boxes, glued_pair_frame_drops
from src.motion.camera_motion import MotionTrack
from src.tracking.appearance import appearance_distance, extract_track_appearance
from src.tracking.bytetrack_boxes import bytetrack_boxes
from src.tracking.enhanced_stitcher import stitch_csv_enhanced
from src.tracking.offfield_filter import filter_offfield_tracks_csv
from src.tracking.smoothing import smooth_tracks_csv
from src.workflow import _video_fps

CLIPS = ["match_center", "match_left", "match_right"]
IMG_W, IMG_H = 1920, 1080

# greedy tracker: same parameters as the first x-model evaluation round
MAX_DIST = 90.0
GAP_ALLOWANCE = 4.0     # extra px of match radius per frame of gap
TRACK_BUFFER = 30       # frames a track survives unseen

# stage-2 appearance veto: clearly different-looking tracks are never merged
VETO_DISTANCE = 0.35


def load_label_boxes(labels_dir: Path) -> dict[int, list[Box]]:
    frames: dict[int, list[Box]] = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        fid = int(txt.stem.rsplit("_", 1)[1])
        boxes = []
        for ln in txt.read_text().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            cls = int(p[0])
            cx, cy, w, h = (float(v) for v in p[1:])
            boxes.append(Box(cls, (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H,
                             (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H))
        frames[fid] = boxes
    return frames


def greedy_track(frames: dict[int, list[Box]]) -> pd.DataFrame:
    """Nearest-centre greedy tracker over person boxes (provisional until a
    small football model runs inside the normal ByteTrack pipeline)."""
    rows = []
    active: dict[int, tuple[float, float, int]] = {}  # id -> (cx, cy, last_fid)
    next_id = 1
    for fid in sorted(frames):
        persons = [b for b in frames[fid] if b.cls in {1, 2, 3}]
        cands = []
        for tid, (px, py, last) in active.items():
            gap = fid - last
            if gap > TRACK_BUFFER:
                continue
            radius = MAX_DIST + GAP_ALLOWANCE * max(0, gap - 1)
            for bi, b in enumerate(persons):
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d <= radius:
                    cands.append((d, tid, bi))
        cands.sort()
        used_t, used_b = set(), set()
        assign: dict[int, int] = {}
        for d, tid, bi in cands:
            if tid in used_t or bi in used_b:
                continue
            used_t.add(tid)
            used_b.add(bi)
            assign[bi] = tid
        for bi, b in enumerate(persons):
            tid = assign.get(bi)
            if tid is None:
                tid = next_id
                next_id += 1
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            active[tid] = (cx, cy, fid)
            rows.append({"frame": fid, "track_id": tid, "class_id": 0,
                         "class_name": "person", "confidence": 0.9,
                         "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                         "center_x": cx, "center_y": cy,
                         "width": b.x2 - b.x1, "height": b.y2 - b.y1})
        active = {t: v for t, v in active.items() if fid - v[2] <= TRACK_BUFFER}
    return pd.DataFrame(rows)


def _drop_ball_shaped(df: pd.DataFrame, max_ar: float = 1.3, max_h: float = 25.0) -> pd.DataFrame:
    """Remove ball / turf-artifact tracks.

    The x-model occasionally labels the ball (or a turf mark) as a *player*,
    so it slips past the class filter and gets tracked. Such a track is
    persistently small AND wider-than-tall, whereas a real player is always
    tall (height/width >= ~1.5, verified: smallest real player still ar>=1.5).
    Done at TRACK level (median shape) so a real player's occasional squat box
    -- a dive, an occlusion -- never drops them.
    """
    w = (df["x2"] - df["x1"]).clip(lower=1)
    h = df["y2"] - df["y1"]
    stats = df.assign(_ar=h / w, _h=h).groupby("track_id").agg(mar=("_ar", "median"), mh=("_h", "median"))
    ball = set(stats[(stats["mar"] < max_ar) & (stats["mh"] < max_h)].index)
    if ball:
        df = df[~df["track_id"].isin(ball)]
    return df, len(ball)


def _drop_short_far(df: pd.DataFrame, min_frames: int = 10, min_height: float = 45.0):
    """Phase 3: drop short AND small tracks -- isolated far blips that could not
    be stitched into anything. Requiring the track to also be *small* (median box
    height < min_height) protects a near player only briefly in view (who is
    large), so only far, unusable fragments are removed. Returns (df, n_dropped).
    """
    h = df["y2"] - df["y1"]
    stats = df.assign(_h=h).groupby("track_id").agg(n=("frame", "size"), mh=("_h", "median"))
    drop = set(stats[(stats["n"] < min_frames) & (stats["mh"] < min_height)].index)
    if drop:
        df = df[~df["track_id"].isin(drop)]
    return df, len(drop)


def rebuild_clip(clip: str, tracker: str = "greedy", prefix: str = "x_",
                 motion_stitch: bool = False) -> None:
    """Rebuild x-model tracking for one clip.

    ``tracker`` selects the ID assignment: "greedy" (provisional centroid)
    or "bytetrack" (Kalman + Hungarian, association in the stabilized
    reference frame). ``prefix`` names the output CSVs so variants can live
    side by side (e.g. "x_" vs "xbt_") for comparison without clobbering.
    """
    root = Path(".")
    det = root / "data" / "output" / clip / "detections"
    labels_dir = root / "data" / "autolabel" / clip / "labels"
    video = root / "data" / "raw" / "test1" / f"{clip}.mp4"
    config = load_config(root / "config" / f"test1_{clip}.yaml")
    motion_cache = det / "camera_motion.npz"

    if prefix == "x_":
        backup = det / "pre_dedup_backup"
        backup.mkdir(exist_ok=True)
        for name in ["x_raw.csv", "x_pitch.csv", "x_tracks.csv", "x_teams.csv"]:
            src = det / name
            if src.exists() and not (backup / name).exists():
                shutil.copy2(src, backup / name)

    print(f"\n=== {clip} [{tracker}] ===")
    frames = load_label_boxes(labels_dir)
    n_before = sum(len([b for b in bs if b.cls in {1, 2, 3}]) for bs in frames.values())

    # stage 1: per-frame fragment / near-identical removal, fully logged
    log_rows = []
    deduped: dict[int, list[Box]] = {}
    for fid, boxes in frames.items():
        kept, removed = dedup_frame_boxes(boxes)
        deduped[fid] = kept
        for rm, keep, rule in removed:
            log_rows.append({"frame": fid, "rule": rule,
                             "rx1": rm.x1, "ry1": rm.y1, "rx2": rm.x2, "ry2": rm.y2,
                             "kx1": keep.x1, "ky1": keep.y1, "kx2": keep.x2, "ky2": keep.y2})
    n_stage1 = len(log_rows)

    if tracker == "bytetrack":
        motion = MotionTrack.load(motion_cache) if motion_cache.exists() else None
        df = bytetrack_boxes(deduped, tracker_cfg=config.tracking.tracker,
                             motion_track=motion)
    else:
        df = greedy_track(deduped)

    # stage 2: glued track pairs (episodic same-person doubles), appearance veto
    drops = glued_pair_frame_drops(df)
    vetoed = 0
    kill: set[tuple[int, int]] = set()   # (frame, track_id) rows to drop
    if drops:
        fingerprints = extract_track_appearance(video, df, stride=5)
        indexed = {int(t): g.set_index("frame") for t, g in df.groupby("track_id")}
        for (a, b), (frame_drops, h_ratio) in drops.items():
            # appearance veto only for similar-size twins: fragment crops
            # never align region-wise with their full-body parent
            if h_ratio > 0.8:
                dist = appearance_distance(fingerprints.get(a), fingerprints.get(b))
                if dist is not None and dist > VETO_DISTANCE:
                    vetoed += 1
                    continue
            for fid, drop_tid in frame_drops:
                keep_tid = b if drop_tid == a else a
                r = indexed[drop_tid].loc[fid]
                k = indexed[keep_tid].loc[fid]
                log_rows.append({"frame": fid, "rule": "glued-pair",
                                 "rx1": r.x1, "ry1": r.y1, "rx2": r.x2, "ry2": r.y2,
                                 "kx1": k.x1, "ky1": k.y1, "kx2": k.x2, "ky2": k.y2})
                kill.add((fid, drop_tid))
        if kill:
            keys = pd.MultiIndex.from_frame(df[["frame", "track_id"]].astype(int))
            df = df[~keys.isin(kill)]
    n_stage2 = len(kill)

    # drop ball / turf artifacts mislabeled as players (persistently squat+small)
    df, n_ball = _drop_ball_shaped(df)

    raw = det / f"{prefix}raw.csv"
    df.to_csv(raw, index=False)
    with (det / f"{prefix}dedup_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "rule", "rx1", "ry1", "rx2", "ry2",
                                          "kx1", "ky1", "kx2", "ky2"])
        w.writeheader()
        w.writerows(log_rows)

    print(f"Boxen: {n_before} -> {n_before - n_stage1 - n_stage2} "
          f"(Stufe 1: -{n_stage1}, Stufe 2: -{n_stage2} aus {len(drops) - vetoed} "
          f"verklebten Track-Paaren, {vetoed} per Appearance-Veto behalten)")
    print(f"Ball/Artefakt-Tracks entfernt: {n_ball}")
    print(f"Tracks (roh): {df.track_id.nunique()}")

    # downstream: pitch coords -> stitch/smooth -> teams (same as before)
    calibration = PitchCalibration.load(Path(config.field.calibration))
    fps = _video_fps(video)
    pitch = det / f"{prefix}pitch.csv"
    filter_to_pitch_csv(raw, pitch, calibration,
                        margin=config.field.margin, video_path=video,
                        motion_cache=motion_cache)
    # remove off-field people (static edge staff/subs; static far-line spectators)
    # that survive the pitch filter -- far_static_only keeps moving far players.
    filter_offfield_tracks_csv(pitch, pitch, far_static_only=True)
    stitched_tmp = det / f"{prefix}stitched_tmp.csv"
    stitch_csv_enhanced(video, pitch, stitched_tmp, fps=fps, motion_stitch_small=motion_stitch)
    tracks = det / f"{prefix}tracks.csv"
    smooth_tracks_csv(stitched_tmp, tracks, fps=fps)
    stitched_tmp.unlink(missing_ok=True)
    # Phase 3: drop isolated far blips (short + small) that survived stitching
    tdf, n_short = _drop_short_far(pd.read_csv(tracks))
    tdf.to_csv(tracks, index=False)
    print(f"Kurz+fern-Blips entfernt: {n_short}")
    teams = det / f"{prefix}teams.csv"
    classify_teams(video, tracks, teams, n_clusters=2)
    print(f"Tracks (final): {pd.read_csv(teams).track_id.nunique()}")


def main() -> None:
    # ByteTrack (Kalman + Hungarian, associated in the stabilized reference
    # frame) is the default: run-path maps show it produces materially
    # cleaner trajectories than the provisional greedy centroid tracker,
    # which conflated players into straight teleport lines. --greedy keeps
    # the old tracker available for comparison.
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]
    clips = args or CLIPS
    if "--stitch" in flags:
        # non-destructive A/B variant: ByteTrack + motion stitching of small
        # far fragments, written to the xs_ prefix so the review tool can compare.
        for clip in clips:
            rebuild_clip(clip, tracker="bytetrack", prefix="xs_", motion_stitch=True)
    else:
        # Phase 3: motion-stitch of small far fragments is now on by default in the
        # production run (x_), so the tester's main run is the least-fragmented one.
        tracker = "greedy" if "--greedy" in flags else "bytetrack"
        for clip in clips:
            rebuild_clip(clip, tracker=tracker, prefix="x_", motion_stitch=True)


if __name__ == "__main__":
    main()
