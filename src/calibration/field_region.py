"""
Field region: a polygon marking the playable pitch in image space.

Used to discard detections of people outside the field (spectators,
coaches, staff) before tracking analysis. Decoupled from detection:
operates on an already-exported detections CSV.

A detection counts as "on the field" when its foot point
(bottom-center of the bounding box, where the player meets the ground)
lies inside the polygon.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(slots=True)
class FieldRegion:
    """A pitch boundary polygon in pixel coordinates."""

    points: list[tuple[float, float]]

    def contour(self) -> np.ndarray:
        return np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)

    def contains(self, x: float, y: float) -> bool:
        """True if pixel (x, y) lies inside (or on) the polygon."""
        return cv2.pointPolygonTest(self.contour(), (float(x), float(y)), False) >= 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"points": self.points}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FieldRegion":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(points=[tuple(p) for p in data["points"]])


def _foot_point(row: pd.Series) -> tuple[float, float]:
    """Bottom-center of the bounding box = where the player stands."""
    return (row["x1"] + row["x2"]) / 2, row["y2"]


def filter_detections_csv(
    input_csv: Path,
    output_csv: Path,
    region: FieldRegion,
) -> None:
    """
    Keep only detections whose foot point is inside ``region`` and write
    the filtered CSV to ``output_csv``.
    """

    df = pd.read_csv(input_csv)

    inside = df.apply(
        lambda r: region.contains(*_foot_point(r)),
        axis=1,
    )

    kept = df[inside].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_csv, index=False)

    removed = len(df) - len(kept)
    print(
        f"Field filter: kept {len(kept)} / {len(df)} detections "
        f"({removed} outside field removed)"
    )
