"""Run Ultralytics ByteTrack on already-computed detection boxes.

Ultralytics couples ByteTrack to ``model.track()``, which would re-run the
(9 s/frame) x-model. We only need the *tracker*, not the detector: this
module drives ``BYTETracker`` directly with detections we already have
(the deduplicated x-model boxes), so ID assignment costs milliseconds.

Unlike the provisional greedy tracker this brings a Kalman motion model
and Hungarian association, which is what stabilizes IDs through crossings
and brief occlusions. The x-model labels carry no per-box confidence, so a
uniform score is assigned; every detection then lands in ByteTrack's
high-confidence association (the low-confidence recovery stage stays
dormant, which is fine -- the boxes are already clean).

The boxes live in RAW frame coordinates (labels were produced on
unstabilized frames). Camera pan is absorbed by the Kalman velocity term;
downstream ``filter_to_pitch_csv`` still applies the per-frame motion
homography, so tracking in raw space keeps rendering and pitch-mapping
consistent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

from src.motion.camera_motion import MotionTrack

PERSON_CLASSES = {1, 2, 3}  # x-model: goalkeeper / player / referee
_UNIFORM_CONF = 0.9


def _warp_xyxy(box, H: np.ndarray) -> tuple[float, float, float, float]:
    """Map a box's 4 corners through homography H and return the axis-aligned
    bounding box of the result. Warping all corners (not just the centre)
    keeps the box valid under the perspective of a panning camera."""
    corners = np.array([[box.x1, box.y1], [box.x2, box.y1],
                        [box.x2, box.y2], [box.x1, box.y2]], np.float64)
    ones = np.ones((4, 1))
    pts = np.hstack([corners, ones]) @ H.T
    pts = pts[:, :2] / pts[:, 2:3]
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


class _Detections:
    """Minimal Results-like view that ``BYTETracker.update`` accepts.

    Needs ``xywh`` (centre format), ``conf``, ``cls`` and boolean-mask
    indexing. Deliberately exposes no ``xywhr`` so parse_bboxes takes the
    axis-aligned path.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_Detections":
        return _Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


def _frame_detections(persons: list, H: np.ndarray | None) -> _Detections:
    """Build the tracker input. When ``H`` is given, boxes are warped into
    the reference frame (camera pan removed) so the Kalman/IoU association
    sees only true player motion."""
    if not persons:
        z = np.zeros((0, 4), np.float32)
        return _Detections(z, np.zeros(0, np.float32), np.zeros(0, np.float32))
    if H is None:
        xywh = np.array([[(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2,
                          b.x2 - b.x1, b.y2 - b.y1] for b in persons], np.float32)
    else:
        rows = []
        for b in persons:
            wx1, wy1, wx2, wy2 = _warp_xyxy(b, H)
            rows.append([(wx1 + wx2) / 2, (wy1 + wy2) / 2, wx2 - wx1, wy2 - wy1])
        xywh = np.array(rows, np.float32)
    conf = np.full(len(persons), _UNIFORM_CONF, np.float32)
    cls = np.array([b.cls for b in persons], np.float32)
    return _Detections(xywh, conf, cls)


def bytetrack_boxes(
    frames: dict[int, list],
    tracker_cfg: str | Path = "config/trackers/bytetrack_occlusion.yaml",
    motion_track: MotionTrack | None = None,
) -> pd.DataFrame:
    """Track deduplicated per-frame boxes with ByteTrack.

    ``frames`` maps frame_id -> list of ``Box`` (src.detection.dedup.Box).
    Frames are stepped in contiguous order so the Kalman gap timing
    (track_buffer counts *frames*) matches real time even across frames
    that momentarily hold no person.

    When ``motion_track`` is given, association runs in the stabilized
    reference frame (camera pan removed) but the STORED coordinates are the
    original RAW boxes: ByteTrack returns the source detection index, so the
    assigned id is re-attached to the untransformed box. Downstream
    ``filter_to_pitch_csv`` therefore still sees raw coordinates and applies
    its own motion compensation exactly as before.

    Returns a DataFrame in the same schema as the greedy tracker's x_raw.
    """
    args = IterableSimpleNamespace(**YAML.load(str(tracker_cfg)))
    tracker = BYTETracker(args)

    fids = sorted(frames)
    rows: list[dict] = []
    # iterate the full contiguous span so gaps advance frame_id too
    for fid in range(fids[0], fids[-1] + 1):
        persons = [b for b in frames.get(fid, []) if b.cls in PERSON_CLASSES]
        H = None if motion_track is None else motion_track.to_reference(fid)
        dets = _frame_detections(persons, H)
        out = tracker.update(dets)  # (M, 8): x1,y1,x2,y2,tid,score,cls,idx
        for row in out:
            tid, score, cls_id, idx = row[4], row[5], row[6], int(row[7])
            src = persons[idx]  # original raw box this track was matched to
            rows.append({
                "frame": fid, "track_id": int(tid),
                "class_id": int(cls_id), "class_name": "person",
                "confidence": float(score),
                "x1": float(src.x1), "y1": float(src.y1),
                "x2": float(src.x2), "y2": float(src.y2),
                "center_x": float((src.x1 + src.x2) / 2),
                "center_y": float((src.y1 + src.y2) / 2),
                "width": float(src.x2 - src.x1), "height": float(src.y2 - src.y1),
            })
    return pd.DataFrame(rows)
