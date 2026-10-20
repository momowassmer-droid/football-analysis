"""
Known pitch landmarks in StatsBomb coordinates (120 x 80).

Pitch runs x = 0..120 (goal to goal), y = 0..80 (touchline to touchline).
The 120 x 80 units always span the full pitch -- they are NOT yards or metres.
A standard pitch is ~105 x 68 m; a 99 x 64 m pitch uses the same coordinate
system but every metric dimension (penalty-area depth, circle radius, ...) maps
to a *different* number of pitch units. ``compute_landmarks`` takes the real
dimensions and returns correctly placed coordinates.

All FIFA metric constants used here:
  penalty area: 16.5 m deep, 40.32 m wide (centred on goal)
  goal area:     5.5 m deep, 18.32 m wide (centred on goal)
  penalty spot: 11.0 m from goal line
  centre circle:  9.15 m radius
  penalty arc:    9.15 m radius from penalty spot, outer edge just touches
                  the penalty-area front line
"""

from dataclasses import dataclass
import math

import numpy as np

# FIFA metric constants (unchanged regardless of pitch size)
_PENALTY_DEPTH_M = 16.5
_PENALTY_WIDTH_M = 40.32
_GOAL_AREA_DEPTH_M = 5.5
_GOAL_AREA_WIDTH_M = 18.32
_PENALTY_SPOT_M = 11.0
_CIRCLE_RADIUS_M = 9.15

# Standard pitch that the old hardcoded numbers assumed.
_STD_LENGTH_M = 105.0
_STD_WIDTH_M  = 68.0


@dataclass(slots=True)
class PitchGeometry:
    """
    Everything needed to draw pitch markings for a real pitch of a given size:
    the named landmark points plus the shapes (circle, arcs) that a point
    dict alone can't express. All in StatsBomb units (120 x 80).
    """
    landmarks: dict[str, tuple[float, float]]
    circle_radius_x: float    # centre-circle radius along the length axis
    circle_radius_y: float    # centre-circle radius along the width axis
    penalty_spot_left_x: float
    penalty_spot_right_x: float
    arc_half_chord_y: float   # half-width (y) of the penalty arc where it meets the box


def compute_pitch_geometry(
    pitch_length_m: float = _STD_LENGTH_M,
    pitch_width_m:  float = _STD_WIDTH_M,
) -> PitchGeometry:
    """
    Compute landmark positions and drawable shape parameters for a pitch whose
    real dimensions are *pitch_length_m* x *pitch_width_m* metres.

    All FIFA-mandated distances (penalty-area depth, centre-circle radius, …)
    are fixed in metres; only the scaling to 0-120 / 0-80 units depends on the
    actual pitch size. Passing the default 105 x 68 m reproduces the old
    hardcoded values. The circle and arc are ellipses in pitch units whenever
    the pitch isn't a 3:2 (120:80) ratio, since x and y are scaled by
    different metre-to-unit factors -- drawing them as a fixed-radius circle
    (as earlier code did) is itself part of what made a non-standard pitch's
    overlay look wrong even with perfectly clicked calibration points.
    """
    SB_L, SB_W = 120.0, 80.0
    sx = SB_L / pitch_length_m   # pitch units per metre, x direction
    sy = SB_W / pitch_width_m    # pitch units per metre, y direction

    # penalty-area corners in StatsBomb units
    pd  = _PENALTY_DEPTH_M  * sx
    phw = _PENALTY_WIDTH_M  / 2 * sy   # half-width from centre

    # goal-area corners
    gd  = _GOAL_AREA_DEPTH_M  * sx
    ghw = _GOAL_AREA_WIDTH_M  / 2 * sy

    # penalty spot
    ps = _PENALTY_SPOT_M * sx

    # centre circle: the four easy tangent points on the axes through (60, 40)
    cr_x = _CIRCLE_RADIUS_M * sx
    cr_y = _CIRCLE_RADIUS_M * sy

    # penalty arc: where the arc (radius 9.15 m from spot) meets the
    # penalty-area front line (x = pd from goal line)
    # distance from spot to front line = pd - ps (in pitch units)
    # half-chord at that distance: sqrt(r² - d²) in *pitch y-units*
    # The arc itself is not circular in pitch units when sx ≠ sy, so we
    # express the chord in metres first then scale.
    dist_spot_to_box_m = _PENALTY_DEPTH_M - _PENALTY_SPOT_M   # 5.5 m
    half_chord_m = math.sqrt(max(0.0, _CIRCLE_RADIUS_M**2 - dist_spot_to_box_m**2))
    arc_y = half_chord_m * sy

    cy = SB_W / 2   # 40.0 — pitch centre y

    landmarks = {
        "Eckfahne oben links":   (0.0,       0.0),
        "Eckfahne unten links":  (0.0,       SB_W),
        "Eckfahne oben rechts":  (SB_L,      0.0),
        "Eckfahne unten rechts": (SB_L,      SB_W),

        "Mittellinie oben":  (SB_L / 2, 0.0),
        "Mittellinie unten": (SB_L / 2, SB_W),
        "Mittelpunkt":       (SB_L / 2, cy),

        # Free touchline points: any click ON the line counts (snapped onto it,
        # see ``line_snap_targets``). Nominal positions are placeholders at 1/3
        # and 2/3 of the line; the snap replaces them with the click's true
        # position along the line. They exist because between corner and
        # halfway line a touchline has no *nameable* landmark, yet extra
        # points along it are exactly what pins the line down against click
        # noise on the few named ones.
        "Seitenlinie oben - frei 1":  (SB_L / 3, 0.0),
        "Seitenlinie oben - frei 2":  (2 * SB_L / 3, 0.0),
        "Seitenlinie unten - frei 1": (SB_L / 3, SB_W),
        "Seitenlinie unten - frei 2": (2 * SB_L / 3, SB_W),

        # Centre-circle tangent points on the two pitch axes
        "Mittelkreis oben":   (SB_L / 2, cy - cr_y),
        "Mittelkreis unten":  (SB_L / 2, cy + cr_y),
        "Mittelkreis links":  (SB_L / 2 - cr_x, cy),
        "Mittelkreis rechts": (SB_L / 2 + cr_x, cy),

        # Left penalty area
        "16er links - oben Torlinie": (0.0,  cy - phw),
        "16er links - unten Torlinie":(0.0,  cy + phw),
        "16er links - oben vorne":    (pd,    cy - phw),
        "16er links - unten vorne":   (pd,    cy + phw),
        # Penalty arc meets the front line
        "Strafraumbogen links - oben am 16er":  (pd, cy - arc_y),
        "Strafraumbogen links - Scheitel":      (ps + _CIRCLE_RADIUS_M * sx, cy),
        "Strafraumbogen links - unten am 16er": (pd, cy + arc_y),
        # Goal area
        "5er links - oben Torlinie":  (0.0, cy - ghw),
        "5er links - unten Torlinie": (0.0, cy + ghw),
        "5er links - oben vorne":     (gd,  cy - ghw),
        "5er links - unten vorne":    (gd,  cy + ghw),
        "Elfmeterpunkt links":        (ps,  cy),

        # Right penalty area (mirror of left)
        "16er rechts - oben Torlinie":  (SB_L,       cy - phw),
        "16er rechts - unten Torlinie": (SB_L,       cy + phw),
        "16er rechts - oben vorne":     (SB_L - pd,  cy - phw),
        "16er rechts - unten vorne":    (SB_L - pd,  cy + phw),
        "Strafraumbogen rechts - oben am 16er":  (SB_L - pd, cy - arc_y),
        "Strafraumbogen rechts - Scheitel":      (SB_L - ps - _CIRCLE_RADIUS_M * sx, cy),
        "Strafraumbogen rechts - unten am 16er": (SB_L - pd, cy + arc_y),
        "5er rechts - oben Torlinie":  (SB_L,      cy - ghw),
        "5er rechts - unten Torlinie": (SB_L,      cy + ghw),
        "5er rechts - oben vorne":     (SB_L - gd, cy - ghw),
        "5er rechts - unten vorne":    (SB_L - gd, cy + ghw),
        "Elfmeterpunkt rechts":        (SB_L - ps, cy),
    }

    return PitchGeometry(
        landmarks=landmarks,
        circle_radius_x=cr_x,
        circle_radius_y=cr_y,
        penalty_spot_left_x=ps,
        penalty_spot_right_x=SB_L - ps,
        arc_half_chord_y=arc_y,
    )


def compute_landmarks(
    pitch_length_m: float = _STD_LENGTH_M,
    pitch_width_m:  float = _STD_WIDTH_M,
) -> dict[str, tuple[float, float]]:
    """Landmark points only; see ``compute_pitch_geometry`` for the full shape data."""
    return compute_pitch_geometry(pitch_length_m, pitch_width_m).landmarks


# Default (standard pitch) -- used wherever pitch dimensions are unknown.
PITCH_LANDMARKS: dict[str, tuple[float, float]] = compute_landmarks()


def nearest_point_on_ellipse(
    px: float, py: float, cx: float, cy: float, rx: float, ry: float, samples: int = 3600,
) -> tuple[float, float]:
    """
    Closest point on an axis-aligned ellipse (centre cx,cy, radii rx,ry) to
    (px, py), via a dense angular grid search.

    The centre-circle's left/right vertex and the penalty-arc's apex are each
    a single mathematically exact point on a curve, but under perspective
    distortion that exact point is generally *not* the pixel that looks like
    the ellipse's widest point in the image -- there's no way to click it
    precisely by eye. This lets a landmark be defined by "closest point on a
    known curve" instead of one exact pixel: whatever the user clicks near
    that curve gets projected onto the true ellipse, so accuracy is only
    needed perpendicular to the curve (easy), not along it (hard). One click
    is cheap (not a per-frame operation), so a plain grid search is precise
    enough without needing an iterative solver.
    """
    if rx <= 0 or ry <= 0:
        return cx, cy
    t = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    ex = cx + rx * np.cos(t)
    ey = cy + ry * np.sin(t)
    d2 = (ex - px) ** 2 + (ey - py) ** 2
    i = int(np.argmin(d2))
    return float(ex[i]), float(ey[i])


def nearest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float,
) -> tuple[float, float]:
    """Closest point on the segment (a, b) to (p) -- projection, clamped."""
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = min(1.0, max(0.0, t))
    return ax + t * dx, ay + t * dy


def line_snap_targets(
    geometry: PitchGeometry,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """
    Landmark name -> pitch-space segment for the "free" line landmarks: a
    click anywhere on that painted line is snapped onto the segment (its
    position along the line comes from the click itself, via the anchor
    homography), so the user needs no identifiable feature -- the line is the
    feature. Same idea as ``circle_snap_targets``, for straight lines.
    """
    return {
        "Seitenlinie oben - frei 1":  ((0.0, 0.0), (120.0, 0.0)),
        "Seitenlinie oben - frei 2":  ((0.0, 0.0), (120.0, 0.0)),
        "Seitenlinie unten - frei 1": ((0.0, 80.0), (120.0, 80.0)),
        "Seitenlinie unten - frei 2": ((0.0, 80.0), (120.0, 80.0)),
    }


def circle_snap_targets(geometry: PitchGeometry) -> dict[str, tuple[float, float, float, float]]:
    """
    Landmark name -> (ellipse_center_x, ellipse_center_y, radius_x, radius_y)
    for the landmarks whose exact position should be found by snapping a
    nearby click onto a known ellipse (see ``nearest_point_on_ellipse``)
    rather than by demanding one precisely-clicked pixel.
    """
    cy = 40.0
    rx, ry = geometry.circle_radius_x, geometry.circle_radius_y
    return {
        "Mittelkreis links":  (60.0, cy, rx, ry),
        "Mittelkreis rechts": (60.0, cy, rx, ry),
        "Strafraumbogen links - Scheitel":  (geometry.penalty_spot_left_x,  cy, rx, ry),
        "Strafraumbogen rechts - Scheitel": (geometry.penalty_spot_right_x, cy, rx, ry),
    }
