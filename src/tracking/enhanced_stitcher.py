"""
Enhanced track stitching in metric pitch space.

Improves on the pixel-distance stitcher by gating fragment merges with
three football-aware signals that only became available once the clip is
calibrated and team colors are sampled:

  1. metric distance  - distance between fragment ends in pitch units
  2. physical speed   - the implied speed must be humanly possible
  3. jersey color     - the two fragments must look like the same player

This directly attacks the "players behind each other" failure: a track
lost behind an opponent is only re-linked to a fragment of the *same*
color that reappears at a physically reachable position.

Requires pitch_x / pitch_y columns (run the calibration filter first).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.tracking.appearance import appearance_distance, extract_track_appearance

# StatsBomb units are ~yards; ~0.9 m each. A sprinter peaks near 11 m/s,
# so ~12 pitch-units/s leaves headroom for calibration noise.
MAX_SPEED_UNITS_PER_SEC = 12.0
MAX_PITCH_DISTANCE = 8.0      # hard cap regardless of gap (units)
# Bhattacharyya distance between two tracks' appearance fingerprints
# (0 identical .. 1 disjoint). 0.5 keeps same-player re-links while rejecting
# a clearly different shirt/shorts combination.
MAX_APPEARANCE_DISTANCE = 0.5
APPEARANCE_STRIDE = 5         # sample one frame in five for the fingerprint
DEFAULT_MAX_FRAME_GAP = 45


# Motion-only fallback (small far fragments that cannot be fingerprinted):
# tighter distance and a same-team guard replace the missing appearance check.
MOTION_MAX_PITCH_DISTANCE = 3.0   # units; << MAX_PITCH_DISTANCE because no appearance guard
FRAGMENT_MAX_LEN = 20             # one side must be this short to allow a motion-only merge


@dataclass(slots=True)
class _Endpoint:
    track_id: int
    start_frame: int
    end_frame: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    team: int
    length: int


def _endpoints(df: pd.DataFrame) -> list[_Endpoint]:
    out = []
    for tid, g in df.sort_values("frame").groupby("track_id"):
        first, last = g.iloc[0], g.iloc[-1]
        team = -1
        if "team_id" in g.columns:
            teams = g["team_id"].dropna()
            if len(teams):
                team = int(teams.mode().iloc[0])
        out.append(_Endpoint(
            track_id=int(tid),
            start_frame=int(first["frame"]),
            end_frame=int(last["frame"]),
            start_xy=(float(first["pitch_x"]), float(first["pitch_y"])),
            end_xy=(float(last["pitch_x"]), float(last["pitch_y"])),
            team=team,
            length=len(g),
        ))
    return out


def _resolve_root(tid: int, links: dict[int, int]) -> int:
    seen: set[int] = set()
    while tid in links and tid not in seen:
        seen.add(tid)
        tid = links[tid]
    return tid


def stitch_csv_enhanced(
    video_path: Path,
    input_csv: Path,
    output_csv: Path,
    fps: float = 30.0,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_speed: float = MAX_SPEED_UNITS_PER_SEC,
    max_appearance_distance: float = MAX_APPEARANCE_DISTANCE,
    class_name: str = "person",
    motion_stitch_small: bool = False,
) -> None:

    df = pd.read_csv(input_csv)
    if "pitch_x" not in df.columns:
        raise ValueError("Enhanced stitcher needs pitch coordinates (run calibration filter first).")

    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"]).copy()
    persons["track_id"] = persons["track_id"].astype(int)

    fingerprints = extract_track_appearance(video_path, persons, stride=APPEARANCE_STRIDE)
    endpoints = _endpoints(persons)
    endpoints.sort(key=lambda e: e.end_frame)

    used: set[int] = set()
    links: dict[int, int] = {}

    for ended in endpoints:
        best, best_cost = None, float("inf")

        for started in endpoints:
            if started.track_id == ended.track_id or started.track_id in used:
                continue
            gap = started.start_frame - ended.end_frame
            if not (0 < gap <= max_frame_gap):
                continue

            dist = float(np.hypot(
                started.start_xy[0] - ended.end_xy[0],
                started.start_xy[1] - ended.end_xy[1],
            ))
            if dist > MAX_PITCH_DISTANCE:
                continue
            if dist / (gap / fps) > max_speed:      # implied speed too high
                continue

            fe, fs = fingerprints.get(ended.track_id), fingerprints.get(started.track_id)
            appear_dist = appearance_distance(fe, fs)
            if appear_dist is None:
                # No appearance fingerprint -- happens for small/far players whose
                # crops are too small to sample. The old code refused these, which
                # left ~50% of the (far-player) fragments unmergeable. Fall back to
                # a MOTION-ONLY merge under stricter guards: much tighter distance,
                # same team, and one side must actually be a short fragment (so two
                # solid tracks never merge on geometry alone).
                if not motion_stitch_small:
                    continue
                if dist > MOTION_MAX_PITCH_DISTANCE:
                    continue
                if min(ended.length, started.length) >= FRAGMENT_MAX_LEN:
                    continue
                if ended.team != -1 and started.team != -1 and ended.team != started.team:
                    continue
                cost = dist + 6.0                    # ranked below any appearance-verified link
            else:
                if appear_dist > max_appearance_distance:  # different-looking player
                    continue
                # Appearance (0..1) scaled to pitch-unit terms so the two costs are
                # comparable; distance breaks ties between similar-looking fragments.
                cost = dist + appear_dist * 10.0
            if cost < best_cost:
                best_cost, best = cost, started.track_id

        if best is not None:
            used.add(best)
            links[best] = ended.track_id

    remap = {tid: _resolve_root(tid, links) for tid in links}
    has_track = df["track_id"].notna()
    df.loc[has_track, "track_id"] = df.loc[has_track, "track_id"].astype(int).replace(remap)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    before = persons["track_id"].nunique()
    after = before - len(links)
    print(f"Enhanced stitch: {before} -> {after} tracks "
          f"({len(links)} merges; metric+speed+color gated)")
