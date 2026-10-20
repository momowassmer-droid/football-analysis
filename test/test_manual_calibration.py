"""Unit tests for the non-GUI parts of manual pitch calibration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.calibration.homography import (
    PitchCalibration,
    compute_manual_calibration,
    image_reprojection_errors,
)
from src.calibration.pitch_landmarks import PITCH_LANDMARKS
from tools.calibrate_field import Viewport, _atomic_save, _quality_warnings


class ViewportTests(unittest.TestCase):
    def test_zoom_keeps_cursor_anchored_to_same_frame_point(self) -> None:
        viewport = Viewport(1920, 1080, 1180, 664)
        cursor = (700, 280)
        before = viewport.display_to_frame(*cursor)

        viewport.zoom_at(2.0, *cursor)
        after = viewport.display_to_frame(*cursor)

        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])

    def test_frame_and_display_coordinates_round_trip(self) -> None:
        viewport = Viewport(1920, 1080, 1180, 664)
        viewport.zoom_at(3.0, 590, 332)

        display = viewport.frame_to_display(960, 540)

        self.assertIsNotNone(display)
        frame = viewport.display_to_frame(*display)
        self.assertAlmostEqual(frame[0], 960, delta=1)
        self.assertAlmostEqual(frame[1], 540, delta=1)


class QualityWarningTests(unittest.TestCase):
    def test_clustered_minimum_points_produce_warnings(self) -> None:
        warnings = _quality_warnings(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            frame_w=1920,
            frame_h=1080,
        )

        self.assertGreaterEqual(len(warnings), 3)


class CalibrationSaveTests(unittest.TestCase):
    def test_atomic_save_remains_compatible_with_calibration_loader(self) -> None:
        calibration = PitchCalibration(
            homography=np.eye(3),
            image_points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            pitch_points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            _atomic_save(calibration, output)
            loaded = PitchCalibration.load(output)

        np.testing.assert_array_equal(loaded.homography, np.eye(3))

    def test_four_manual_points_are_reproduced_exactly(self) -> None:
        pitch = [(0.0, 0.0), (120.0, 0.0), (120.0, 80.0), (0.0, 80.0)]
        image = [(110.0, 90.0), (1810.0, 130.0), (1550.0, 980.0), (260.0, 940.0)]

        calibration = compute_manual_calibration(image, pitch)

        self.assertLess(float(image_reprojection_errors(calibration).max()), 0.01)


class PitchLandmarkTests(unittest.TestCase):
    def test_side_points_of_center_circle_are_not_offered(self) -> None:
        self.assertNotIn("Mittelkreis links", PITCH_LANDMARKS)
        self.assertNotIn("Mittelkreis rechts", PITCH_LANDMARKS)

    def test_goal_area_landmarks_exist_on_both_sides(self) -> None:
        self.assertEqual(PITCH_LANDMARKS["5er links - oben vorne"], (6.0, 30.0))
        self.assertEqual(PITCH_LANDMARKS["5er links - unten vorne"], (6.0, 50.0))
        self.assertEqual(PITCH_LANDMARKS["5er rechts - oben vorne"], (114.0, 30.0))
        self.assertEqual(PITCH_LANDMARKS["5er rechts - unten vorne"], (114.0, 50.0))

    def test_penalty_arc_landmarks_exist_on_both_sides(self) -> None:
        self.assertEqual(
            PITCH_LANDMARKS["Strafraumbogen links - oben am 16er"], (18.0, 32.0)
        )
        self.assertEqual(
            PITCH_LANDMARKS["Strafraumbogen links - Scheitel"], (22.0, 40.0)
        )
        self.assertEqual(
            PITCH_LANDMARKS["Strafraumbogen rechts - oben am 16er"], (102.0, 32.0)
        )
        self.assertEqual(
            PITCH_LANDMARKS["Strafraumbogen rechts - Scheitel"], (98.0, 40.0)
        )


if __name__ == "__main__":
    unittest.main()
