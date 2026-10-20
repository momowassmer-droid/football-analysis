"""
Evaluation metrics for tracking quality.

Reads a detections CSV (as produced by CsvExporter) and computes
simple, interpretable metrics that quantify known weaknesses such as
track fragmentation, ID switches and frame-to-frame detection drops.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.tracking.track_stitcher import find_continuation_links


@dataclass(slots=True)
class TrackingReport:
    """
    Aggregated tracking-quality metrics for a single detections CSV.
    """

    total_detections: int
    total_frames: int
    unique_track_ids: int

    track_lengths: pd.Series
    mean_track_length: float
    median_track_length: float
    short_tracks_ratio: float

    detections_per_frame: pd.Series
    mean_detections_per_frame: float

    suspected_id_switches: int

    fragmentation_links: int
    estimated_real_players: int

    def print_summary(self) -> None:
        print("\n========== TRACKING EVALUATION ==========\n")
        print(f"Total detections       : {self.total_detections}")
        print(f"Total frames            : {self.total_frames}")
        print(f"Unique track IDs        : {self.unique_track_ids}")
        print(f"Mean track length       : {self.mean_track_length:.1f} frames")
        print(f"Median track length     : {self.median_track_length:.1f} frames")
        print(f"Short tracks (<5 fr.)   : {self.short_tracks_ratio:.1%}")
        print(f"Mean detections/frame   : {self.mean_detections_per_frame:.2f}")
        print(f"Suspected ID switches   : {self.suspected_id_switches}")
        print(f"Fragmentation links     : {self.fragmentation_links}")
        print(f"Estimated real players  : {self.estimated_real_players}")
        print(
            f"  (i.e. {self.unique_track_ids} track IDs likely belong to "
            f"only {self.estimated_real_players} physical players)"
        )


SHORT_TRACK_THRESHOLD_FRAMES = 5
ID_SWITCH_JUMP_THRESHOLD_PX = 150.0


def evaluate_tracking(
    detections_csv: Path,
    class_name: str = "person",
) -> TrackingReport:
    """
    Compute tracking-quality metrics for one class (default: "person").
    """

    df = pd.read_csv(detections_csv)

    df = df[df["class_name"] == class_name]
    df = df.dropna(subset=["track_id"])
    df["track_id"] = df["track_id"].astype(int)

    track_lengths = df.groupby("track_id").size()

    detections_per_frame = df.groupby("frame").size()

    short_tracks_ratio = (
        (track_lengths < SHORT_TRACK_THRESHOLD_FRAMES).sum() / len(track_lengths)
        if len(track_lengths) > 0
        else 0.0
    )

    suspected_id_switches = _count_suspected_id_switches(df)
    fragmentation_links = _count_fragmentation_links(df)

    return TrackingReport(
        total_detections=len(df),
        total_frames=df["frame"].nunique(),
        unique_track_ids=df["track_id"].nunique(),
        track_lengths=track_lengths,
        mean_track_length=float(track_lengths.mean()) if len(track_lengths) else 0.0,
        median_track_length=float(track_lengths.median()) if len(track_lengths) else 0.0,
        short_tracks_ratio=float(short_tracks_ratio),
        detections_per_frame=detections_per_frame,
        mean_detections_per_frame=float(detections_per_frame.mean()) if len(detections_per_frame) else 0.0,
        suspected_id_switches=suspected_id_switches,
        fragmentation_links=fragmentation_links,
        estimated_real_players=int(df["track_id"].nunique() - fragmentation_links),
    )


def _count_suspected_id_switches(df: pd.DataFrame) -> int:
    """
    Flag a track as a suspected ID switch when its bounding-box center
    jumps more than ID_SWITCH_JUMP_THRESHOLD_PX between two consecutive
    frames the track appears in. A large, sudden jump usually means the
    tracker re-assigned this ID to a different player.
    """

    switches = 0

    for _, track in df.sort_values("frame").groupby("track_id"):

        centers = track[["x1", "y1", "x2", "y2"]].to_numpy()
        center_x = (centers[:, 0] + centers[:, 2]) / 2
        center_y = (centers[:, 1] + centers[:, 3]) / 2

        if len(center_x) < 2:
            continue

        jumps = np.hypot(np.diff(center_x), np.diff(center_y))

        switches += int((jumps > ID_SWITCH_JUMP_THRESHOLD_PX).sum())

    return switches


def _count_fragmentation_links(df: pd.DataFrame) -> int:
    """
    Count how many track IDs are likely continuations of an earlier track
    that was briefly lost (e.g. due to occlusion) and re-assigned a new ID.
    Reuses the same link-finding the track stitcher uses to merge them, so
    measurement and fix stay in sync.
    """

    return len(find_continuation_links(df))
