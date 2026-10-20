"""
Fully manual pitch calibration.

The tool deliberately performs no automatic line or keypoint detection. You
choose every image landmark yourself; the resulting correspondences are used
only to calculate the image -> pitch homography.

Usage:
    python -m tools.calibrate_field VIDEO OUTPUT_JSON [FRAME_INDEX]

Controls:
    left click       set the current landmark
    mouse wheel      zoom around the cursor
    right-drag       move the zoomed image
    0                reset zoom
    n / b            next (skip) / previous landmark
    u                undo the last recorded point
    v                show or hide the projected pitch preview
    s                preview first, then save after a second press
    q / ESC          quit without saving

Existing calibration files are never touched until the final save. Saving is
atomic, so a reader can never observe a half-written JSON file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.calibration.homography import (
    PitchCalibration,
    compute_manual_calibration,
    image_reprojection_errors,
    suggest_worst_point,
)
from src.calibration.pitch_landmarks import (
    circle_snap_targets,
    compute_pitch_geometry,
    line_snap_targets,
    nearest_point_on_ellipse,
    nearest_point_on_segment,
)
from src.visualization.pitch_overlay import draw_pitch_lines


MAX_WINDOW_W = 1540
MAX_WINDOW_H = 860
PANEL_W = 360
MIN_POINTS = 4
RECOMMENDED_POINTS = 6
MAX_ZOOM = 10.0


@dataclass(slots=True)
class Viewport:
    """Reversible mapping between the full frame and the visible image area."""

    frame_w: int
    frame_h: int
    display_w: int
    display_h: int
    zoom: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.zoom = 1.0
        self.center_x = self.frame_w / 2
        self.center_y = self.frame_h / 2

    def _visible_size(self) -> tuple[float, float]:
        return self.frame_w / self.zoom, self.frame_h / self.zoom

    def _clamp_center(self) -> None:
        visible_w, visible_h = self._visible_size()
        self.center_x = float(np.clip(
            self.center_x, visible_w / 2, self.frame_w - visible_w / 2
        ))
        self.center_y = float(np.clip(
            self.center_y, visible_h / 2, self.frame_h - visible_h / 2
        ))

    def bounds(self) -> tuple[float, float, float, float]:
        visible_w, visible_h = self._visible_size()
        x0 = self.center_x - visible_w / 2
        y0 = self.center_y - visible_h / 2
        return x0, y0, x0 + visible_w, y0 + visible_h

    def display_to_frame(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds()
        return (
            x0 + (x / self.display_w) * (x1 - x0),
            y0 + (y / self.display_h) * (y1 - y0),
        )

    def frame_to_display(self, x: float, y: float) -> tuple[int, int] | None:
        x0, y0, x1, y1 = self.bounds()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        return (
            int(round((x - x0) / (x1 - x0) * self.display_w)),
            int(round((y - y0) / (y1 - y0) * self.display_h)),
        )

    def zoom_at(self, factor: float, display_x: float, display_y: float) -> None:
        anchor_x, anchor_y = self.display_to_frame(display_x, display_y)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, MAX_ZOOM))
        visible_w, visible_h = self._visible_size()
        self.center_x = anchor_x + (0.5 - display_x / self.display_w) * visible_w
        self.center_y = anchor_y + (0.5 - display_y / self.display_h) * visible_h
        self._clamp_center()

    def pan(self, display_dx: float, display_dy: float) -> None:
        visible_w, visible_h = self._visible_size()
        self.center_x -= display_dx / self.display_w * visible_w
        self.center_y -= display_dy / self.display_h * visible_h
        self._clamp_center()

    def render(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds()
        left, top = int(np.floor(x0)), int(np.floor(y0))
        right, bottom = int(np.ceil(x1)), int(np.ceil(y1))
        crop = frame[top:bottom, left:right]
        return cv2.resize(crop, (self.display_w, self.display_h),
                          interpolation=cv2.INTER_LINEAR)


def _quality_warnings(
    image_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
) -> list[str]:
    """Return human-readable warnings; weak geometry remains saveable."""

    warnings: list[str] = []
    if len(image_points) < RECOMMENDED_POINTS:
        warnings.append(f"Only {len(image_points)} points; {RECOMMENDED_POINTS}+ recommended")

    image = np.asarray(image_points, dtype=np.float64)
    pitch = np.asarray(pitch_points, dtype=np.float64)
    if len(image):
        image_span_x = float(np.ptp(image[:, 0]) / max(frame_w, 1))
        image_span_y = float(np.ptp(image[:, 1]) / max(frame_h, 1))
        if image_span_x < 0.20 or image_span_y < 0.15:
            warnings.append("Clicks are clustered in the image")
    if len(pitch):
        pitch_span_x = float(np.ptp(pitch[:, 0]))
        pitch_span_y = float(np.ptp(pitch[:, 1]))
        if pitch_span_x < 30 or pitch_span_y < 20:
            warnings.append("Landmarks cover too little of the pitch")
    return warnings


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (230, 230, 230),
    scale: float = 0.43,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _atomic_save(calibration: PitchCalibration, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_name(f".{output_json.name}.tmp")
    try:
        calibration.save(temporary)
        temporary.replace(output_json)
    finally:
        if temporary.exists():
            temporary.unlink()


def calibrate_video(
    video_path: Path,
    output_json: Path,
    frame_index: int = 0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> bool:
    """Open the fully manual calibration UI and report whether it was saved."""

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read frame {frame_index} from {video_path}")
        return False

    frame_h, frame_w = frame.shape[:2]
    image_max_w = MAX_WINDOW_W - PANEL_W
    scale = min(image_max_w / frame_w, MAX_WINDOW_H / frame_h, 1.0)
    display_w = max(1, int(round(frame_w * scale)))
    display_h = max(1, int(round(frame_h * scale)))
    viewport = Viewport(frame_w, frame_h, display_w, display_h)

    geometry = compute_pitch_geometry(pitch_length_m, pitch_width_m)
    landmarks = geometry.landmarks
    snap_targets = circle_snap_targets(geometry)
    line_targets = line_snap_targets(geometry)
    names = list(landmarks)
    idx = 0
    recorded: dict[str, tuple[float, float]] = {}
    order: list[str] = []
    preview: PitchCalibration | None = None
    warnings: list[str] = []
    point_errors: dict[str, float] = {}
    status = "Select every visible landmark yourself"

    dragging = False
    last_drag = (0, 0)
    window = "Manual pitch calibration"

    def selected_points() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        return (
            [recorded[name] for name in order],
            [landmarks[name] for name in order],
        )

    def snap_circle_points(
        image_points: list[tuple[float, float]],
        pitch_points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Replace the nominal target of a curve/line landmark with the point on
        its true shape nearest to where the user actually clicked: ellipse
        vertices (``snap_targets``) and free touchline points
        (``line_targets``). Needs >= 4 regular anchor points to locate the
        shapes first; until then the nominal targets are used unchanged.
        """
        free = [i for i, n in enumerate(order) if n in snap_targets or n in line_targets]
        if not free:
            return pitch_points
        anchors = [i for i in range(len(order)) if i not in free]
        if len(anchors) < 4:
            return pitch_points
        anchor_cal = compute_manual_calibration(
            [image_points[i] for i in anchors], [pitch_points[i] for i in anchors],
            verbose=False,
        )
        result = list(pitch_points)
        for i in free:
            px, py = anchor_cal.image_to_pitch(*image_points[i])
            name = order[i]
            if name in snap_targets:
                cx, cy, rx, ry = snap_targets[name]
                result[i] = nearest_point_on_ellipse(px, py, cx, cy, rx, ry)
            else:
                (ax, ay), (bx, by) = line_targets[name]
                result[i] = nearest_point_on_segment(px, py, ax, ay, bx, by)
        return result

    def make_preview() -> bool:
        nonlocal preview, warnings, point_errors, status
        if len(recorded) < MIN_POINTS:
            status = f"Need at least {MIN_POINTS} points; currently {len(recorded)}"
            return False
        image_points, pitch_points = selected_points()
        pitch_points = snap_circle_points(image_points, pitch_points)
        try:
            candidate = compute_manual_calibration(image_points, pitch_points)
            # Also validates that the matrix is invertible for the overlay.
            np.linalg.inv(candidate.homography)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error) as exc:
            preview = None
            warnings = ["Geometry is degenerate; choose wider-spread points"]
            point_errors = {}
            status = f"Preview failed: {exc}"
            return False
        preview = candidate
        warnings = _quality_warnings(
            image_points, pitch_points, frame_w=frame_w, frame_h=frame_h
        )
        point_errors = dict(zip(order, image_reprojection_errors(candidate)))
        # Leave-one-out advice: a single inconsistent click (wrong line / a
        # mis-placed far corner) smears its error across the whole overlay,
        # and no fit can compensate -- so point the user straight at the click
        # worth re-placing, with the error it would save. Advisory only: the
        # point stays until the user re-clicks or skips it (this is a fully
        # manual tool; nothing is auto-rejected).
        suggestion = suggest_worst_point(image_points, pitch_points)
        if suggestion is not None:
            drop_index, improved, current = suggestion
            # Require a clear improvement, not just whichever point happens to
            # be the marginal worst: on real (slightly imperfect) clicks, the
            # leave-one-out delta tails off smoothly (e.g. 15.5 -> 8.9 -> 2.2px
            # on a 10-point set) and only the first one or two are genuine
            # mis-clicks. A small delta is normal fit noise, not a bad click --
            # flagging it would send the user chasing a point that was fine.
            #
            # This is a hint, not a verdict: it names the single click whose
            # removal would help most *right now*, not a claim that the user
            # mis-clicked it. The user judges against what they actually see
            # in the frame, and every fresh preview (this function reruns on
            # each click/undo) recomputes the suggestion from scratch -- so if
            # they decide the named point is fine and adjust a different one
            # instead, the next preview will reflect that, not repeat the same
            # accusation.
            if current > 6.0 and improved < current - 6.0:
                worst_name = order[drop_index].split(" - ")[0]
                warnings.insert(
                    0, f"Largest inconsistency now at {worst_name} "
                       f"(max {current:.0f}->{improved:.0f}px if resolved) - "
                       f"check it against the frame"
                )
        elif point_errors:
            worst_name, worst_error = max(point_errors.items(), key=lambda item: item[1])
            if worst_error > 3.0:
                warnings.insert(0, f"Largest mismatch {worst_error:.0f}px: {worst_name}")
        status = "Preview active: inspect yellow lines; S again saves"
        return True

    def invalidate_preview(message: str) -> None:
        nonlocal preview, warnings, point_errors, status
        preview = None
        warnings = []
        point_errors = {}
        status = message

    def redraw() -> None:
        source = frame.copy()
        if preview is not None:
            draw_pitch_lines(source, preview.homography, pitch_length_m, pitch_width_m)

        image = viewport.render(source)
        canvas = np.zeros((display_h, display_w + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :display_w] = image
        canvas[:, display_w:] = (32, 32, 32)

        for name, (full_x, full_y) in recorded.items():
            point = viewport.frame_to_display(full_x, full_y)
            if point is None:
                continue
            error = point_errors.get(name, 0.0)
            if preview is None or error <= 3.0:
                color = (0, 255, 0)
            elif error <= 8.0:
                color = (0, 170, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 9, (0, 0, 0), 1, cv2.LINE_AA)
            error_label = f" {error:.0f}px" if preview is not None else ""
            _draw_text(canvas, name.split(" - ")[0] + error_label,
                       point[0] + 9, point[1] - 8,
                       color=color, scale=0.38)

        panel_x = display_w + 14
        _draw_text(canvas, "MANUAL PITCH CALIBRATION", panel_x, 27,
                   color=(0, 255, 255), scale=0.55, thickness=2)
        _draw_text(canvas, "No automatic field detection", panel_x, 49,
                   color=(150, 200, 255))
        _draw_text(canvas, f"Frame: {frame_index}    Zoom: {viewport.zoom:.1f}x",
                   panel_x, 76)
        _draw_text(canvas, f"Points: {len(recorded)} / {len(names)}", panel_x, 98)

        current = names[idx] if idx < len(names) else "All landmarks visited"
        _draw_text(canvas, "CURRENT:", panel_x, 128, color=(180, 180, 180))
        _draw_text(canvas, current, panel_x, 151,
                   color=(0, 255, 255), scale=0.48, thickness=2)
        if current in snap_targets or current in line_targets:
            _draw_text(canvas, "Klick muss nicht exakt sein -",
                       panel_x, 168, color=(0, 200, 255), scale=0.38)
            _draw_text(canvas, "nur IRGENDWO auf der Linie",
                       panel_x, 183, color=(0, 200, 255), scale=0.38)
            status_y = 202
        else:
            status_y = 181
        _draw_text(canvas, status[:43], panel_x, status_y, color=(120, 220, 255))
        if len(status) > 43:
            _draw_text(canvas, status[43:86], panel_x, status_y + 19, color=(120, 220, 255))

        y = 229
        if warnings:
            _draw_text(canvas, "QUALITY WARNINGS", panel_x, y,
                       color=(0, 170, 255), thickness=2)
            y += 22
            for warning in warnings[:3]:
                _draw_text(canvas, f"- {warning[:40]}", panel_x, y,
                           color=(0, 170, 255))
                y += 20
            y += 5

        _draw_text(canvas, "Nearby landmarks", panel_x, y,
                   color=(180, 180, 180), thickness=2)
        y += 22
        start = max(0, min(idx - 3, len(names) - 7))
        for pos in range(start, min(start + 7, len(names))):
            marker = ">" if pos == idx else ("x" if names[pos] in recorded else " ")
            color = (0, 255, 255) if pos == idx else (150, 220, 150)
            _draw_text(canvas, f"{marker} {names[pos][:36]}", panel_x, y, color=color)
            y += 20

        controls_y = max(y + 12, display_h - 145)
        _draw_text(canvas, "CONTROLS", panel_x, controls_y,
                   color=(180, 180, 180), thickness=2)
        for line in (
            "click=set   n=skip   b=back   u=undo",
            "wheel=zoom   right-drag=pan   0=reset",
            "v=preview   s=preview/save   q=quit",
        ):
            controls_y += 22
            _draw_text(canvas, line, panel_x, controls_y)

        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        nonlocal idx, dragging, last_drag
        if event == cv2.EVENT_RBUTTONUP:
            dragging = False
            return
        if x >= display_w or y >= display_h:
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            direction = 1 if flags > 0 else -1
            viewport.zoom_at(1.35 if direction > 0 else 1 / 1.35, x, y)
            redraw()
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            dragging = True
            last_drag = (x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and dragging:
            viewport.pan(x - last_drag[0], y - last_drag[1])
            last_drag = (x, y)
            redraw()
            return
        if event == cv2.EVENT_LBUTTONDOWN and idx < len(names):
            name = names[idx]
            if name not in recorded:
                order.append(name)
            recorded[name] = viewport.display_to_frame(x, y)
            idx += 1
            invalidate_preview(f"Set: {name}")
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    saved = False
    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("0"):
            viewport.reset()
            redraw()
        elif key == ord("n") and idx < len(names):
            idx += 1
            invalidate_preview("Skipped landmark")
            redraw()
        elif key == ord("b") and idx > 0:
            idx -= 1
            invalidate_preview("Moved to previous landmark")
            redraw()
        elif key == ord("u") and order:
            last = order.pop()
            recorded.pop(last, None)
            idx = names.index(last)
            invalidate_preview(f"Removed: {last}")
            redraw()
        elif key == ord("v"):
            if preview is None:
                make_preview()
            else:
                preview = None
                status = "Preview hidden; points unchanged"
            redraw()
        elif key == ord("s"):
            if preview is None:
                make_preview()
                redraw()
                continue
            _atomic_save(preview, output_json)
            print(f"Saved manual calibration ({len(recorded)} points) to {output_json}")
            saved = True
            break

    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass  # already closed via the window manager
    return saved


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    frame_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    config_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    pitch_length_m, pitch_width_m = 105.0, 68.0
    if config_path is not None and config_path.exists():
        from src.config.config_loader import load_config
        cfg = load_config(config_path)
        pitch_length_m = cfg.venue.pitch_length_m
        pitch_width_m  = cfg.venue.pitch_width_m
        print(f"Pitch dimensions from config: {pitch_length_m} x {pitch_width_m} m")

    calibrate_video(Path(sys.argv[1]), Path(sys.argv[2]),
                    frame_index, pitch_length_m, pitch_width_m)


if __name__ == "__main__":
    main()
