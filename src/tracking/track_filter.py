"""
Track-level validity filter (needs pitch coordinates).

Per-frame field filtering tolerates a margin around the pitch so players
standing on a line are not lost because of small calibration errors. At
track level, repeated foot points outside the actual touch-/goal-lines
identify off-pitch people without making assumptions about movement.

Operates on a detections CSV that already has pitch_x / pitch_y columns
(from PitchCalibration). Non-person rows pass through untouched.
"""

from pathlib import Path

import pandas as pd


def filter_tracks_csv(
    input_csv: Path,
    output_csv: Path,
    pitch_margin: float = 0.5,
    min_inside_ratio: float = 0.60,
    class_name: str = "person",
) -> None:
    """
    Drop person tracks when fewer than ``min_inside_ratio`` of their foot
    points are within the pitch boundaries. ``pitch_margin`` only absorbs
    small homography/calibration errors at the lines. Movement is not used.
    """

    df = pd.read_csv(input_csv)

    if "pitch_x" not in df.columns or "pitch_y" not in df.columns:
        raise ValueError("Track filter needs pitch_x/pitch_y; run the calibration filter first.")

    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"])

    keep_ids: set[int] = set()
    dropped: list[tuple[int, float]] = []

    for track_id, g in persons.groupby("track_id"):
        inside = (
            g["pitch_x"].between(-pitch_margin, 120 + pitch_margin)
            & g["pitch_y"].between(-pitch_margin, 80 + pitch_margin)
        )
        inside_ratio = float(inside.mean())

        if inside_ratio < min_inside_ratio:
            dropped.append((int(track_id), inside_ratio))
        else:
            keep_ids.add(int(track_id))

    is_person = df["class_name"] == class_name
    keep_mask = (~is_person) | df["track_id"].isin(keep_ids)
    kept = df[keep_mask].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_csv, index=False)

    print(
        f"Sideline filter: kept {len(keep_ids)} tracks, dropped {len(dropped)} "
        f"off-pitch tracks (minimum inside ratio: {min_inside_ratio:.0%})"
    )
