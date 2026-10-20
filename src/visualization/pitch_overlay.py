"""
Draw StatsBomb pitch lines onto a frame from an image->pitch homography.

Each pitch line is sampled densely and projected pitch->image. Points that
fall behind the camera horizon (homogeneous w <= 0) or at absurd image
coordinates are dropped, and only runs of consecutive valid points are
drawn. This prevents the old failure where a line touching the horizon was
drawn as a single straight segment shooting across the frame.

Takes the image->pitch homography directly so a per-frame (motion
compensated) homography can be passed.

The segment geometry (penalty area, goal area, centre circle) is derived
from real metric pitch dimensions via ``pitch_landmarks.compute_landmarks``,
the same function the calibration tool uses. Earlier this module hardcoded
the segment coordinates for a standard 105 x 68 m pitch, so on a
differently-sized pitch (common for club/youth grounds) the calibration
itself could be correct while this overlay still drew the penalty area and
centre circle at the wrong size -- a visual mismatch that had nothing to do
with how well the landmarks were clicked.
"""

import cv2
import numpy as np

from src.calibration.pitch_landmarks import compute_pitch_geometry

LINE_COLOR = (0, 255, 255)
LINE_THICKNESS = 2
_SAMPLES_PER_SEGMENT = 60


def field_line_segments(landmarks: dict[str, tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Straight pitch-line segments, in StatsBomb coordinates (120 x 80)."""
    L = landmarks
    return [
        [(0, 0), (120, 0)], [(120, 0), (120, 80)],
        [(120, 80), (0, 80)], [(0, 80), (0, 0)],
        [(60, 0), (60, 80)],                                  # halfway line

        [L["16er links - oben Torlinie"], L["16er links - oben vorne"]],
        [L["16er links - oben vorne"], L["16er links - unten vorne"]],
        [L["16er links - unten vorne"], L["16er links - unten Torlinie"]],

        [L["16er rechts - oben Torlinie"], L["16er rechts - oben vorne"]],
        [L["16er rechts - oben vorne"], L["16er rechts - unten vorne"]],
        [L["16er rechts - unten vorne"], L["16er rechts - unten Torlinie"]],

        [L["5er links - oben Torlinie"], L["5er links - oben vorne"]],
        [L["5er links - oben vorne"], L["5er links - unten vorne"]],
        [L["5er links - unten vorne"], L["5er links - unten Torlinie"]],

        [L["5er rechts - oben Torlinie"], L["5er rechts - oben vorne"]],
        [L["5er rechts - oben vorne"], L["5er rechts - unten vorne"]],
        [L["5er rechts - unten vorne"], L["5er rechts - unten Torlinie"]],
    ]


def _project_to_image(points_pitch: np.ndarray, pitch_to_image: np.ndarray,
                      w_img: int, h_img: int):
    """
    Project pitch points to image. A point is valid if it lands within one
    image-size beyond the frame edges; horizon-crossing points blow up to
    huge coordinates and are rejected, which breaks the polyline there
    instead of drawing a segment shooting across the frame.
    """
    src = points_pitch.reshape(-1, 1, 2).astype(np.float32)
    coords = cv2.perspectiveTransform(src, pitch_to_image).reshape(-1, 2)
    valid = (
        (coords[:, 0] > -w_img) & (coords[:, 0] < 2 * w_img)
        & (coords[:, 1] > -h_img) & (coords[:, 1] < 2 * h_img)
    )
    return coords, valid


def _draw_polyline_valid(image: np.ndarray, coords: np.ndarray, valid: np.ndarray) -> None:
    """Draw connected line segments only between consecutive valid points."""
    for i in range(len(coords) - 1):
        if valid[i] and valid[i + 1]:
            p1 = tuple(np.round(coords[i]).astype(int))
            p2 = tuple(np.round(coords[i + 1]).astype(int))
            cv2.line(image, p1, p2, LINE_COLOR, LINE_THICKNESS)


def draw_pitch_lines(
    image: np.ndarray,
    image_to_pitch: np.ndarray,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> None:
    """
    Overlay pitch markings using a (possibly per-frame) image->pitch homography.

    Pass the venue's real pitch dimensions so the penalty area, goal area and
    centre circle are drawn at their true size for this specific pitch -- the
    defaults (105 x 68 m) only match a standard-sized pitch.
    """
    geometry = compute_pitch_geometry(pitch_length_m, pitch_width_m)
    pitch_to_image = np.linalg.inv(image_to_pitch)
    h_img, w_img = image.shape[:2]

    for a, b in field_line_segments(geometry.landmarks):
        ts = np.linspace(0, 1, _SAMPLES_PER_SEGMENT)
        seg = np.array([(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in ts])
        coords, valid = _project_to_image(seg, pitch_to_image, w_img, h_img)
        _draw_polyline_valid(image, coords, valid)

    # center circle: an ellipse in pitch units whenever the pitch isn't 3:2
    cr_x, cr_y = geometry.circle_radius_x, geometry.circle_radius_y
    angles = np.linspace(0, 2 * np.pi, 48)
    circle = np.stack([60 + cr_x * np.cos(angles), 40 + cr_y * np.sin(angles)], axis=1)
    coords, valid = _project_to_image(circle, pitch_to_image, w_img, h_img)
    _draw_polyline_valid(image, coords, valid)
    if valid[0] and valid[-1]:                      # close the loop
        cv2.line(image, tuple(np.round(coords[-1]).astype(int)),
                 tuple(np.round(coords[0]).astype(int)), LINE_COLOR, LINE_THICKNESS)

    # Penalty arcs: same ellipse radii as the centre circle (FIFA uses the
    # same 9.15 m radius for both), centred on each penalty spot. Only the
    # section outside the penalty area is painted; it meets the box front
    # line at y = 40 +/- arc_half_chord_y. Solve the ellipse angle range
    # directly from that known half-chord: y = 40 + cr_y*sin(t)
    # => sin(t) = arc_half_chord_y / cr_y.
    t_max = np.arcsin(np.clip(geometry.arc_half_chord_y / cr_y, -1.0, 1.0)) if cr_y else 0.0
    angles = np.linspace(-t_max, t_max, 32)
    for center_x, direction in (
        (geometry.penalty_spot_left_x, 1.0),
        (geometry.penalty_spot_right_x, -1.0),
    ):
        arc = np.stack([
            center_x + direction * cr_x * np.cos(angles),
            40 + cr_y * np.sin(angles),
        ], axis=1)
        coords, valid = _project_to_image(arc, pitch_to_image, w_img, h_img)
        _draw_polyline_valid(image, coords, valid)
