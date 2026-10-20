"""
Ball trajectory extraction from multi-class YOLO detections.

The generic COCO model used for player detection also emits a "sports
ball" class, but on small/fast-moving footage it flickers and sometimes
misclassifies the ball as a visually similar class (observed: "frisbee").
Rather than training a dedicated ball detector, this module exploits a
property a player tracker can't: there is exactly one ball, so among all
ball-like detections in a frame the right one is whichever is closest to
the last known ball position.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BALL_CLASSES = ("sports ball", "frisbee")
MAX_BALL_JUMP_PX = 250.0  # discard a candidate that implies an implausible jump


def extract_ball_track(
    detections_csv: Path,
    output_csv: Path,
    ball_classes: tuple[str, ...] = DEFAULT_BALL_CLASSES,
    max_jump_px: float = MAX_BALL_JUMP_PX,
) -> pd.DataFrame:
    """
    Reduce all ball-like detections to at most one ball position per
    frame and write a compact ball-only CSV (frame, x, y, confidence).
    """

    df = pd.read_csv(detections_csv)
    candidates = df[df["class_name"].isin(ball_classes)].copy()
    candidates["x"] = (candidates["x1"] + candidates["x2"]) / 2
    candidates["y"] = (candidates["y1"] + candidates["y2"]) / 2

    by_frame = {fid: g for fid, g in candidates.groupby("frame")}

    rows = []
    last_pos: tuple[float, float] | None = None

    for frame_id in range(int(df["frame"].max()) + 1):

        group = by_frame.get(frame_id)
        if group is None or len(group) == 0:
            continue

        if last_pos is None:
            best = group.loc[group["confidence"].idxmax()]
        else:
            distances = np.hypot(group["x"] - last_pos[0], group["y"] - last_pos[1])
            group = group.assign(_distance=distances)
            group = group[group["_distance"] <= max_jump_px]
            if len(group) == 0:
                continue
            best = group.loc[group["_distance"].idxmin()]

        last_pos = (float(best["x"]), float(best["y"]))

        rows.append(
            {
                "frame": frame_id,
                "x": best["x"],
                "y": best["y"],
                "confidence": best["confidence"],
                "class_name": best["class_name"],
            }
        )

    ball_track = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ball_track.to_csv(output_csv, index=False)

    total_frames = int(df["frame"].max()) + 1
    print(
        f"Ball track: {len(ball_track)}/{total_frames} frames "
        f"({len(ball_track) / total_frames:.0%} coverage)"
    )

    return ball_track
