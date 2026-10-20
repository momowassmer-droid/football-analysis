"""
Pitch calibration via homography: image pixels -> StatsBomb coordinates.

Built from >= 4 point correspondences (clicked landmark pixel <-> known
pitch coordinate). Decoupled from detection/tracking: the resulting
homography is applied to an already-exported detections CSV.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(slots=True)
class PitchCalibration:
    """Homography mapping image pixels to pitch coordinates (120 x 80)."""

    homography: np.ndarray  # 3x3
    image_points: list[tuple[float, float]]
    pitch_points: list[tuple[float, float]]

    def image_to_pitch(self, x: float, y: float) -> tuple[float, float]:
        """Map a single image pixel to pitch coordinates."""
        src = np.array([[[float(x), float(y)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.homography)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "homography": self.homography.tolist(),
                    "image_points": self.image_points,
                    "pitch_points": self.pitch_points,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "PitchCalibration":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            homography=np.array(data["homography"], dtype=np.float64),
            image_points=[tuple(p) for p in data["image_points"]],
            pitch_points=[tuple(p) for p in data["pitch_points"]],
        )


def compute_calibration(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
) -> PitchCalibration:
    """
    Compute the image->pitch homography from >= 4 correspondences.
    Uses RANSAC so a single mis-clicked point doesn't ruin the fit.
    """

    if len(image_points) < 4:
        raise ValueError(
            f"Need at least 4 correspondences, got {len(image_points)}"
        )

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(pitch_points, dtype=np.float32)

    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if homography is None:
        raise RuntimeError("Homography could not be computed (degenerate points?)")

    calibration = PitchCalibration(
        homography=homography,
        image_points=image_points,
        pitch_points=pitch_points,
    )
    report_calibration_quality(calibration)
    return calibration


def image_reprojection_errors(calibration: "PitchCalibration") -> np.ndarray:
    """Pixel distance between each clicked point and its projected landmark."""

    pitch_to_image = np.linalg.inv(calibration.homography)
    pitch = np.asarray(calibration.pitch_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pitch, pitch_to_image).reshape(-1, 2)
    clicked = np.asarray(calibration.image_points, dtype=np.float64)
    return np.linalg.norm(projected - clicked, axis=1)


def _max_pixel_error(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
) -> float:
    """Worst clicked-point pixel error of the all-points pitch->image fit."""

    image = np.asarray(image_points, dtype=np.float32)
    pitch = np.asarray(pitch_points, dtype=np.float32)
    pitch_to_image, _ = cv2.findHomography(pitch, image, method=0)
    if pitch_to_image is None:
        return float("inf")
    projected = cv2.perspectiveTransform(
        pitch.reshape(-1, 1, 2), pitch_to_image
    ).reshape(-1, 2)
    return float(np.max(np.linalg.norm(projected - image, axis=1)))


def suggest_worst_point(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
) -> tuple[int, float, float] | None:
    """
    Leave-one-out: the click whose removal most lowers the worst pixel error.

    A homography is the optimal single projective map for the clicks, so it
    cannot be improved by a better algorithm -- only by better clicks. One
    point that landed on the wrong line (easy on a pitch with overlaid
    markings) or imprecisely on a far corner is inconsistent with the rest,
    and the global fit smears that inconsistency across the whole overlay.
    Refitting without each point in turn surfaces the offender: we return the
    one whose removal most reduces the maximum reprojection error, so the user
    can re-click (or skip) just that landmark instead of guessing. Returns
    ``(index, max_error_without_it, current_max_error)`` or None when there are
    too few points to drop one.
    """

    n = len(image_points)
    if n <= 4:  # 4 points define the homography exactly; can't drop one
        return None
    current = _max_pixel_error(image_points, pitch_points)
    best_index, best_error = 0, float("inf")
    for drop in range(n):
        keep = [i for i in range(n) if i != drop]
        error = _max_pixel_error(
            [image_points[i] for i in keep], [pitch_points[i] for i in keep]
        )
        if error < best_error:
            best_index, best_error = drop, error
    return best_index, best_error, current


def compute_manual_calibration(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
    verbose: bool = True,
) -> PitchCalibration:
    """
    Fit a homography for manually clicked landmarks without rejecting points.

    The normal calibration fits image -> pitch and uses RANSAC, which is useful
    for automatic detections but may silently ignore a manual click and can
    create a visually large overlay offset near the horizon. Manual calibration
    instead fits pitch -> image with ordinary least squares, then inverts it.
    Consequently every click participates and the visible pixel error is the
    quantity minimized. Four non-collinear correspondences are reproduced
    exactly; with more points this is the closest possible single homography.
    """

    if len(image_points) != len(pitch_points):
        raise ValueError("Image and pitch point counts must match")
    if len(image_points) < 4:
        raise ValueError(f"Need at least 4 correspondences, got {len(image_points)}")

    image = np.asarray(image_points, dtype=np.float32)
    pitch = np.asarray(pitch_points, dtype=np.float32)
    pitch_to_image, _ = cv2.findHomography(pitch, image, method=0)
    if pitch_to_image is None:
        raise RuntimeError("Homography could not be computed (degenerate points?)")

    try:
        image_to_pitch = np.linalg.inv(pitch_to_image)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Homography is not invertible (degenerate points?)") from exc

    calibration = PitchCalibration(
        homography=image_to_pitch,
        image_points=image_points,
        pitch_points=pitch_points,
    )
    if verbose:
        errors = image_reprojection_errors(calibration)
        print(
            "Manual calibration visual error: "
            f"mean {float(np.mean(errors)):.1f} px, max {float(np.max(errors)):.1f} px "
            "(all clicks included)"
        )
    return calibration


def report_calibration_quality(calibration: "PitchCalibration") -> float:
    """
    Print the mean reprojection error (pitch units) and warn if the marked
    points are clustered. Well-distributed points across the whole visible
    field give a homography that stays accurate at the edges, not just the
    center. Returns the mean error.
    """

    errors = []
    for ip, pp in zip(calibration.image_points, calibration.pitch_points):
        mx, my = calibration.image_to_pitch(*ip)
        errors.append(float(np.hypot(mx - pp[0], my - pp[1])))
    mean_error = float(np.mean(errors)) if errors else 0.0

    pitch = np.array(calibration.pitch_points)
    spread_x = pitch[:, 0].max() - pitch[:, 0].min() if len(pitch) else 0
    spread_y = pitch[:, 1].max() - pitch[:, 1].min() if len(pitch) else 0

    print(f"Calibration quality: mean reprojection error = {mean_error:.2f} pitch units")
    print(f"  point spread: {spread_x:.0f} x {spread_y:.0f} of 120 x 80")
    if spread_x < 60 or spread_y < 40:
        print("  WARNING: points are clustered - add far corners / opposite side "
              "for an accurate edge-to-edge homography.")
    return mean_error


PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def _map_foot_points(df: pd.DataFrame, calibration: PitchCalibration) -> np.ndarray:
    """Map each detection's foot point to pitch coordinates; returns (N, 2)."""
    foot_x = (df["x1"] + df["x2"]) / 2
    foot_y = df["y2"]
    pts = np.stack([foot_x.to_numpy(), foot_y.to_numpy()], axis=1)
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, calibration.homography).reshape(-1, 2)


def add_pitch_coordinates_csv(
    input_csv: Path,
    output_csv: Path,
    calibration: PitchCalibration,
) -> None:
    """
    Add pitch_x / pitch_y columns to a detections CSV by mapping each
    detection's foot point (bottom-center of the box) through the
    homography.
    """

    df = pd.read_csv(input_csv)
    mapped = _map_foot_points(df, calibration)

    df["pitch_x"] = mapped[:, 0]
    df["pitch_y"] = mapped[:, 1]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Added pitch coordinates to {len(df)} detections -> {output_csv}")


def _vertical_sensitivity(foot_x: np.ndarray, foot_y: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Pitch-units of movement per 1px of *vertical* foot-point movement, per point.

    Near the horizon a homography compresses the whole far half of the pitch
    into a thin band of image rows, so a foot point there moves metres on the
    pitch for a single pixel of detection jitter -- unusable, and the source of
    phantom sprints along the far touchline. Measured by finite difference
    (foot_y +/- 0.5 px) through the same per-frame homography.
    """
    up = np.stack([foot_x, foot_y - 0.5], axis=1).reshape(-1, 1, 2).astype(np.float32)
    dn = np.stack([foot_x, foot_y + 0.5], axis=1).reshape(-1, 1, 2).astype(np.float32)
    pu = cv2.perspectiveTransform(up, h).reshape(-1, 2)
    pd_ = cv2.perspectiveTransform(dn, h).reshape(-1, 2)
    return np.hypot(pu[:, 0] - pd_[:, 0], pu[:, 1] - pd_[:, 1])


def _map_foot_points_compensated(
    df: pd.DataFrame, video_path: Path, calibration: PitchCalibration, motion_cache: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like ``_map_foot_points``, but composes a per-frame image->pitch
    homography from the cached camera-motion track instead of reusing one
    static homography for the whole clip. A calibration is only measured on
    one reference frame; if the (virtual) camera pans or zooms during the
    clip, every detection inherits that camera motion as spurious pitch
    movement -- e.g. players "running" in big curves that are really just
    the camera drifting. Returns pitch coordinates in the same row order as
    ``df``. Pure array math -- no video decode needed once the track is cached.

    The per-frame transform comes from ``camera_motion`` (a full projective
    homography per frame, anchored to the reference frame), which is markedly
    more accurate over a long pan than the earlier 4-DOF affine motion log --
    the composition is a single matmul because both factors are already 3x3.
    """
    from src.motion.camera_motion import camera_motion_for

    foot_x = (df["x1"] + df["x2"]) / 2
    foot_y = df["y2"]
    by_frame: dict[int, list[int]] = {}
    for pos, fid in enumerate(df["frame"].to_numpy()):
        by_frame.setdefault(int(fid), []).append(pos)

    track = camera_motion_for(video_path, motion_cache)
    out = np.full((len(df), 2), np.nan, dtype=np.float64)
    sens = np.full(len(df), np.nan, dtype=np.float64)
    fx, fy = foot_x.to_numpy(), foot_y.to_numpy()
    for fid, positions in by_frame.items():
        h = calibration.homography @ track.to_reference(fid)
        pts = np.stack([fx[positions], fy[positions]], axis=1).reshape(-1, 1, 2).astype(np.float32)
        out[positions] = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
        sens[positions] = _vertical_sensitivity(fx[positions], fy[positions], h)
    return out, sens


def filter_to_pitch_csv(
    input_csv: Path,
    output_csv: Path,
    calibration: PitchCalibration,
    margin: float = 2.0,
    video_path: Path | None = None,
    motion_cache: Path | None = None,
    max_sensitivity: float | None = None,
) -> None:
    """
    Keep only detections whose foot point maps inside the pitch
    (0..120 x 0..80, expanded by ``margin`` to tolerate players right on
    the touchline and small calibration error). Also adds pitch_x /
    pitch_y columns, so the output already carries StatsBomb coordinates.

    ``max_sensitivity`` (pitch-units per vertical pixel) drops detections in
    the horizon-compressed far band, where a single pixel of foot-point jitter
    maps to metres on the pitch -- those positions are unusable and produce
    phantom far-touchline sprints. 0.25 keeps the entire real pitch (mid-field
    is ~0.04, the far line ~0.5) while removing the unreliable strip. Only
    applied on the motion-compensated path (needs the per-frame homography);
    set to None to disable.

    Pass ``video_path`` to compensate for camera motion (composing a
    per-frame homography) instead of applying one static homography to the
    whole clip -- strongly recommended whenever the source camera pans or
    zooms, which is the norm for Veo-style footage. ``motion_cache`` points
    at the clip's shared motion-log file, so this reuses an alignment the
    detector's stabilizer already computed (or computes and saves one for
    the renderer to reuse) instead of estimating it again from scratch.
    """

    df = pd.read_csv(input_csv)
    sens: np.ndarray | None = None
    if video_path is not None:
        mapped, sens = _map_foot_points_compensated(df, video_path, calibration, motion_cache)
    else:
        mapped = _map_foot_points(df, calibration)
    df["pitch_x"] = mapped[:, 0]
    df["pitch_y"] = mapped[:, 1]

    inside = (
        (df["pitch_x"] >= -margin)
        & (df["pitch_x"] <= PITCH_LENGTH + margin)
        & (df["pitch_y"] >= -margin)
        & (df["pitch_y"] <= PITCH_WIDTH + margin)
    )
    n_horizon = 0
    if sens is not None and max_sensitivity is not None:
        reliable = sens <= max_sensitivity
        n_horizon = int((inside & ~reliable).sum())
        inside = inside & reliable
    kept = df[inside].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_csv, index=False)

    removed = len(df) - len(kept)
    mode = "motion-compensated" if video_path is not None else "static"
    horizon_note = f", {n_horizon} horizon-unreliable" if n_horizon else ""
    print(
        f"Pitch filter ({mode} calibration): kept {len(kept)} / {len(df)} "
        f"({removed - n_horizon} off-pitch{horizon_note} removed); added pitch coordinates"
    )
