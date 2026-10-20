"""
CSV exporter for detections.
"""

from pathlib import Path

import pandas as pd

from src.core.detection import Detection


class CsvExporter:

    @staticmethod
    def export_detections(
        detections: list[Detection],
        output_path: Path,
    ) -> None:

        rows = []

        for d in detections:

            rows.append(
                {
                    "frame": d.frame_id,
                    "track_id": d.track_id,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "x1": d.x1,
                    "y1": d.y1,
                    "x2": d.x2,
                    "y2": d.y2,
                    "center_x": d.center_x,
                    "center_y": d.center_y,
                    "width": d.width,
                    "height": d.height,
                }
            )

        df = pd.DataFrame(rows)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)