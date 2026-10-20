"""
Team classification via jersey color clustering.

Decoupled post-processing step: reads an existing detections CSV plus
the source video, samples each detection's jersey color from the torso
region of its bounding box, and clusters tracks into teams. No new
detector or model needed -- just the already-tracked boxes.

Color is aggregated per track (not per detection) before clustering, so
a track keeps one consistent team_id even if lighting/occlusion makes a
single frame's color sample noisy.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.video.video_reader import VideoReader

# Torso sampling window as a fraction of the bounding box: avoids the
# head (color varies: skin/hair) and the legs/shorts (often a different,
# less team-distinctive color than the shirt).
TORSO_X_RANGE = (0.30, 0.70)
TORSO_Y_RANGE = (0.15, 0.55)

REFEREE_TEAM_ID = 3  # color-outlier tracks (rendered distinctly, not a team)


def _torso_color(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """
    Median BGR jersey color of the torso region with grass masked out.

    For small/distant players the box is mostly pitch, so unmasked sampling
    returns grass green and the player ends up in a spurious "green team".
    Pixels whose hue is green and saturated are dropped; if too few jersey
    pixels remain the sample is rejected (None) rather than trusted.
    """

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    tx1 = int(x1 + w * TORSO_X_RANGE[0])
    tx2 = int(x1 + w * TORSO_X_RANGE[1])
    ty1 = int(y1 + h * TORSO_Y_RANGE[0])
    ty2 = int(y1 + h * TORSO_Y_RANGE[1])

    region = frame[max(ty1, 0):max(ty2, 1), max(tx1, 0):max(tx2, 1)]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    grass = (hue >= 35) & (hue <= 90) & (sat > 60)
    jersey = region[~grass]

    if len(jersey) < max(10, 0.15 * region[:, :, 0].size):
        return None  # almost all grass -> unreliable, skip this sample

    return np.median(jersey, axis=0)


def _extract_track_colors(
    video_path: Path,
    df: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Sample torso colors for every detection, return median color per track_id."""

    by_frame = {fid: g for fid, g in df.groupby("frame")}
    samples: dict[int, list[np.ndarray]] = {}

    reader = VideoReader(video_path)
    reader.open()

    for frame in reader.frames():

        rows = by_frame.get(frame.frame_id)
        if rows is None:
            continue

        for _, row in rows.iterrows():
            color = _torso_color(frame.image, row["x1"], row["y1"], row["x2"], row["y2"])
            if color is None:
                continue
            samples.setdefault(int(row["track_id"]), []).append(color)

    reader.close()

    return {
        track_id: np.median(np.stack(colors), axis=0)
        for track_id, colors in samples.items()
    }


def _bgr_to_cluster_features(colors_bgr: np.ndarray) -> np.ndarray:
    """
    Convert per-track mean BGR colors to HSV clustering features.

    Hue and saturation carry jersey identity; value (brightness) separates
    white from dark kits. Hue is scaled down so a single noisy hue on a
    low-saturation (white/grey) shirt cannot dominate the distance.
    """
    bgr = colors_bgr.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] *= 0.5   # downweight hue (0..180) relative to S and V (0..255)
    return hsv


@dataclass(slots=True)
class TeamAssignment:
    track_colors: dict[int, np.ndarray]
    track_team: dict[int, int]
    cluster_centers: np.ndarray  # (n_clusters, 3) BGR


def classify_teams(
    video_path: Path,
    detections_csv: Path,
    output_csv: Path,
    class_name: str = "person",
    n_clusters: int = 2,
    detect_referees: bool = False,
    referee_factor: float = 1.8,
    referee_track_ids: list[int] | None = None,
) -> TeamAssignment:
    """
    Cluster person tracks into ``n_clusters`` teams by jersey color.

    With ``detect_referees`` the two teams are clustered first, then tracks
    whose color sits far from *both* team centers (distance > referee_factor
    x the median) are relabelled as referee (team_id REFEREE_TEAM_ID) instead
    of being forced into a team. This only works if the referee's kit is
    actually a distinct color; if it matches a team, color cannot separate it.
    """

    df = pd.read_csv(detections_csv)
    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"]).copy()
    persons["track_id"] = persons["track_id"].astype(int)

    track_colors = _extract_track_colors(video_path, persons)

    track_ids = list(track_colors.keys())
    colors = np.array([track_colors[t] for t in track_ids], dtype=np.float32)

    # Cluster in HSV rather than BGR: jersey identity lives in hue/saturation/
    # brightness, which separates white vs dark vs colored kits far better
    # than raw BGR euclidean (which is dominated by lighting).
    features = _bgr_to_cluster_features(colors)

    n_clusters = min(n_clusters, len(track_ids)) if track_ids else 0

    if n_clusters < 2:
        track_team = {t: 0 for t in track_ids}
        centers = colors[:1] if len(colors) else np.zeros((0, 3), dtype=np.float32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, feat_centers = cv2.kmeans(
            features, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        track_team = {t: int(label) for t, label in zip(track_ids, labels)}

        if detect_referees:
            dist_to_centers = np.linalg.norm(
                features[:, None, :] - feat_centers[None, :, :], axis=2
            )
            nearest = dist_to_centers.min(axis=1)
            median_d = float(np.median(nearest)) if len(nearest) else 0.0
            for i, tid in enumerate(track_ids):
                if median_d > 0 and nearest[i] > referee_factor * median_d:
                    track_team[tid] = REFEREE_TEAM_ID

        # manual override: reliable when color cannot separate the referee
        for tid in (referee_track_ids or []):
            if tid in track_team:
                track_team[tid] = REFEREE_TEAM_ID

        # report centers back in BGR for readability
        centers = np.array([
            colors[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(3)
            for c in range(n_clusters)
        ], dtype=np.float32)

    df["team_id"] = df["track_id"].map(
        lambda tid: track_team.get(int(tid)) if pd.notna(tid) else None
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Classified {len(track_ids)} tracks into {n_clusters} teams.")
    for cluster_id in range(n_clusters):
        b, g, r = centers[cluster_id]
        members = [t for t, c in track_team.items() if c == cluster_id]
        print(f"  Team {cluster_id}: {len(members)} tracks, mean BGR=({b:.0f},{g:.0f},{r:.0f})")
    n_ref = sum(1 for c in track_team.values() if c == REFEREE_TEAM_ID)
    if n_ref:
        print(f"  Referee/outliers: {n_ref} tracks (distinct color, not a team)")

    return TeamAssignment(
        track_colors=track_colors,
        track_team=track_team,
        cluster_centers=centers,
    )
