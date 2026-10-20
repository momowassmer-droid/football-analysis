"""
Interactive field-boundary marker.

Opens a reference frame from a video and lets you click the corners of
the pitch (in order, e.g. far-left, far-right, near-right, near-left).
The polygon is saved as JSON for use with FieldRegion.

Usage:
    python -m tools.mark_field data/raw/spiel2_clip.mp4 config/fields/spiel2_clip.json
    python -m tools.mark_field <video> <output_json> [frame_index]

Controls:
    left click : add a corner
    u          : undo last corner
    s          : save and quit
    q / ESC    : quit without saving
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from src.calibration.field_region import FieldRegion


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return

    video_path = Path(sys.argv[1])
    output_json = Path(sys.argv[2])
    frame_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"Could not read frame {frame_index} from {video_path}")
        return

    points: list[tuple[float, float]] = []

    def redraw() -> None:
        canvas = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) >= 2:
            pts = [(int(x), int(y)) for x, y in points]
            cv2.polylines(canvas, [np.array(pts)], len(points) > 2, (0, 255, 0), 2)
        cv2.imshow("Mark field (click corners, s=save, u=undo, q=quit)", canvas)

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
            redraw()

    cv2.namedWindow("Mark field (click corners, s=save, u=undo, q=quit)")
    cv2.setMouseCallback("Mark field (click corners, s=save, u=undo, q=quit)", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == ord("s") and len(points) >= 3:
            FieldRegion(points=points).save(output_json)
            print(f"Saved {len(points)} points to {output_json}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
