"""Render motion-compensated pitch calibration and stabilization diagnostics.

The three test views use a manual calibration measured on frame 0.  The
camera-motion track maps every later frame back to that reference, so the
per-frame image-to-pitch homography is::

    calibration_frame = calibration_reference @ frame_to_reference

This script renders the requested calibration frames at full resolution,
creates one contact sheet per view, and writes a side-by-side stabilization
video for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.calibration.homography import PitchCalibration
from src.motion.camera_motion import IDENTITY_H, MotionTrack, _PhotometricVerifier
from src.visualization.pitch_overlay import draw_pitch_lines


PITCH_LENGTH_M = 99.0
PITCH_WIDTH_M = 64.0
DEFAULT_FRAMES = (100, 200, 300, 400, 500, 599)


@dataclass(frozen=True)
class View:
    name: str
    video: Path
    calibration: Path
    motion: Path


VIEWS = (
    View(
        "left",
        Path("data/raw/test1/match_left.mp4"),
        Path("config/fields/test1_match_left_calib_v2.json"),
        Path("data/output/match_left/detections/camera_motion.npz"),
    ),
    View(
        "center",
        Path("data/raw/test1/match_center.mp4"),
        Path("config/fields/test1_match_center_calib_v3.json"),
        Path("data/output/match_center/detections/camera_motion.npz"),
    ),
    View(
        "right",
        Path("data/raw/test1/match_right.mp4"),
        Path("config/fields/test1_match_right_calib_v2.json"),
        Path("data/output/match_right/detections/camera_motion.npz"),
    ),
)


def _label(image: np.ndarray, text: str, x: int = 24, y: int = 44) -> None:
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    return capture


def _read_frame(capture: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id}")
    return frame


def _overlay(frame: np.ndarray, homography: np.ndarray, label: str) -> np.ndarray:
    result = frame.copy()
    draw_pitch_lines(result, homography, PITCH_LENGTH_M, PITCH_WIDTH_M)
    _label(result, label)
    return result


def _contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    thumbs: list[np.ndarray] = []
    for image in images:
        thumb = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    return np.vstack((np.hstack(thumbs[:3]), np.hstack(thumbs[3:])))


def _stabilization_contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    thumbs = [
        cv2.resize(image, (960, 270), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    return np.vstack((
        np.hstack(thumbs[0:2]),
        np.hstack(thumbs[2:4]),
        np.hstack(thumbs[4:6]),
    ))


def _comparison_frame(
    frame: np.ndarray,
    frame_id: int,
    calibration: PitchCalibration,
    transform: np.ndarray,
) -> np.ndarray:
    dynamic_h = calibration.homography @ transform
    original = _overlay(frame, dynamic_h, f"Original - Frame {frame_id}")

    h, w = frame.shape[:2]
    stabilized = cv2.warpPerspective(frame, transform, (w, h))
    stabilized = _overlay(
        stabilized, calibration.homography, f"Stabilisiert auf Frame 0 - Frame {frame_id}"
    )

    original = cv2.resize(original, (960, 540), interpolation=cv2.INTER_AREA)
    stabilized = cv2.resize(stabilized, (960, 540), interpolation=cv2.INTER_AREA)
    return np.hstack((original, stabilized))


def render_view(view: View, output_root: Path, frame_ids: tuple[int, ...]) -> dict:
    calibration = PitchCalibration.load(view.calibration)
    track = MotionTrack.load(view.motion)
    capture = _open_video(view.video)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if len(track.transforms) != frame_count:
        raise RuntimeError(
            f"{view.name}: motion track has {len(track.transforms)} transforms "
            f"for {frame_count} video frames"
        )
    if any(frame_id < 0 or frame_id >= frame_count for frame_id in frame_ids):
        raise ValueError(f"{view.name}: requested frame outside 0..{frame_count - 1}")

    view_dir = output_root / view.name
    view_dir.mkdir(parents=True, exist_ok=True)
    reference = _read_frame(capture, track.reference_frame_id)
    verifier = _PhotometricVerifier(reference)

    rendered: list[np.ndarray] = []
    stabilization_comparisons: list[np.ndarray] = []
    metrics: list[dict] = []
    for frame_id in frame_ids:
        frame = _read_frame(capture, frame_id)
        transform = track.to_reference(frame_id)
        dynamic_h = calibration.homography @ transform
        image = _overlay(frame, dynamic_h, f"{view.name.upper()} - Frame {frame_id}")
        output_path = view_dir / f"frame_{frame_id:03d}_calibration.jpg"
        if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {output_path}")
        rendered.append(image)
        stabilization_comparisons.append(
            _comparison_frame(frame, frame_id, calibration, transform)
        )

        gray = verifier.downscale(frame)
        state = track.camera_state(frame_id)
        raw_error = verifier.error(gray, IDENTITY_H)
        stabilized_error = verifier.error(gray, transform)
        metrics.append({
            "frame": frame_id,
            "pan_x_px": round(state.dx, 3),
            "pan_y_px": round(state.dy, 3),
            "zoom": round(state.zoom, 6),
            "rotation_deg": round(state.rotation_deg, 4),
            "alignment_error_raw": round(raw_error, 3),
            "alignment_error_stabilized": round(stabilized_error, 3),
        })

    sheet_path = output_root / f"{view.name}_calibration_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), _contact_sheet(rendered),
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    stabilization_sheet_path = output_root / f"{view.name}_stabilization_contact_sheet.jpg"
    cv2.imwrite(
        str(stabilization_sheet_path),
        _stabilization_contact_sheet(stabilization_comparisons),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    comparison_path = output_root / f"{view.name}_stabilization_comparison.mp4"
    writer = cv2.VideoWriter(
        str(comparison_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 540)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {comparison_path}")
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(_comparison_frame(
            frame, frame_id, calibration, track.to_reference(frame_id)
        ))
        frame_id += 1
    writer.release()
    capture.release()
    if frame_id != frame_count:
        raise RuntimeError(f"{view.name}: rendered {frame_id}/{frame_count} video frames")

    metrics_path = view_dir / "stabilization_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer_csv.writeheader()
        writer_csv.writerows(metrics)

    improvements = [
        row["alignment_error_raw"] - row["alignment_error_stabilized"]
        for row in metrics
    ]
    return {
        "view": view.name,
        "video_frames": frame_count,
        "motion_frames": len(track.transforms),
        "reference_frame": track.reference_frame_id,
        "anchored_keyframes": track.anchored_keyframes,
        "hold_frames": track.hold_frames,
        "median_camera_displacement_px": round(track.median_displacement(), 3),
        "mean_selected_alignment_improvement": round(float(np.mean(improvements)), 3),
        "rendered_frames": list(frame_ids),
        "contact_sheet": str(sheet_path),
        "stabilization_contact_sheet": str(stabilization_sheet_path),
        "comparison_video": str(comparison_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/output/field_stabilization_test"),
    )
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [render_view(view, args.output, tuple(args.frames)) for view in VIEWS]
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
