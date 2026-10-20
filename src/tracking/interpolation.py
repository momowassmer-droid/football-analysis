"""
Fill occlusion gaps inside a track by interpolation.

After stitching, a single player's track can still have holes: frames
where the detector lost them behind another player. Since the track now
has one consistent ID across the gap, the missing frames can be filled
by linearly interpolating between the surrounding detections. This turns
fragmented tracks into continuous trajectories suitable for run paths and
possession analysis.

Interpolated rows are flagged with interpolated=1 so downstream analysis
can tell measured positions from filled ones.
"""

from pathlib import Path

import numpy as np
import pandas as pd

INTERP_COLUMNS = ["x1", "y1", "x2", "y2", "pitch_x", "pitch_y"]
DEFAULT_MAX_GAP = 30


def interpolate_tracks_csv(
    input_csv: Path,
    output_csv: Path,
    max_gap: int = DEFAULT_MAX_GAP,
    class_name: str = "person",
) -> None:
    """
    For every person track, insert interpolated rows for missing frames
    in gaps up to ``max_gap`` frames long. Larger gaps are left as holes
    (too long to interpolate reliably).
    """

    df = pd.read_csv(input_csv)
    if "interpolated" not in df.columns:
        df["interpolated"] = 0

    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"])

    new_rows = []

    for tid, g in persons.groupby("track_id"):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy()

        for i in range(len(frames) - 1):
            f0, f1 = int(frames[i]), int(frames[i + 1])
            gap = f1 - f0
            if gap <= 1 or gap > max_gap:
                continue

            row0, row1 = g.iloc[i], g.iloc[i + 1]
            for f in range(f0 + 1, f1):
                t = (f - f0) / gap
                filled = {
                    "frame": f,
                    "track_id": tid,
                    "class_id": row0["class_id"],
                    "class_name": class_name,
                    "confidence": 0.0,
                    "interpolated": 1,
                }
                for col in INTERP_COLUMNS:
                    if col in g.columns:
                        filled[col] = (1 - t) * row0[col] + t * row1[col]
                new_rows.append(filled)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Interpolation: filled {len(new_rows)} occlusion-gap frames")
