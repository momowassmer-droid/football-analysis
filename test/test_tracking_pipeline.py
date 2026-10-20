"""Regression tests for tracking configuration and pitch-line filtering."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.config.config_loader import load_config
from src.core.frame import Frame
from src.detection.yolo_detector import YoloDetector
from src.tracking.track_filter import filter_tracks_csv


class TrackingConfigurationTests(unittest.TestCase):
    def test_detection_confidence_is_forwarded_to_yolo(self) -> None:
        with patch("src.detection.yolo_detector.YOLO") as yolo_class:
            model = MagicMock()
            yolo_class.return_value = model
            detector = YoloDetector(confidence=0.37)
            detector.detect(Frame(frame_id=0, timestamp=0.0, image=MagicMock()))

        self.assertEqual(model.track.call_args.kwargs["conf"], 0.37)

    def test_config_loads_sideline_filter_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                """
video: {input: clip.mp4}
output: {detections: detections.csv}
track_filter:
  enabled: true
  pitch_margin: 0.4
  min_inside_ratio: 0.7
""",
                encoding="utf-8",
            )
            config = load_config(config_path)

        self.assertTrue(config.track_filter.enabled)
        self.assertEqual(config.track_filter.pitch_margin, 0.4)
        self.assertEqual(config.track_filter.min_inside_ratio, 0.7)


class SidelineFilterTests(unittest.TestCase):
    def test_filter_uses_pitch_boundaries_not_movement(self) -> None:
        rows = []
        for frame in range(10):
            rows.append(_person(frame, 1, 60.0, 40.0))  # static but on-pitch
            rows.append(_person(frame, 2, 60.0, -1.2))  # static and off-pitch

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "tracks.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            filter_tracks_csv(csv_path, csv_path, pitch_margin=0.5, min_inside_ratio=0.6)
            filtered = pd.read_csv(csv_path)

        self.assertEqual(set(filtered["track_id"]), {1})
        self.assertEqual(len(filtered), 10)


def _person(frame: int, track_id: int, pitch_x: float, pitch_y: float) -> dict[str, object]:
    return {
        "frame": frame,
        "track_id": track_id,
        "class_name": "person",
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
    }


if __name__ == "__main__":
    unittest.main()
