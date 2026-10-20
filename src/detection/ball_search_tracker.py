"""
Hebel 1: search-window ball tracking.

Once the ball's position is roughly known, the next position is nearby. So
instead of detecting the tiny ball in the full 1920px frame, this crops a
small window around the Kalman prediction and runs the detector on just that
crop. In the crop the ball occupies far more of the input, so its effective
resolution is much higher and recall jumps -- exactly in the frames the
full-frame pass misses.

Flow per frame: Kalman predict -> crop around prediction -> detect in crop
(low confidence) -> pick the ball nearest the prediction -> update, else fall
back to a global candidate, else coast. Initialized from global candidates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from src.detection.ball_lowconf import BALL_COCO_IDS
from src.detection.ball_tracker import _kalman_matrices
from src.video.video_reader import VideoReader

GATE_PX = 120.0
MAX_COAST_FRAMES = 25


STATIC_RADIUS_PX = 35.0
STATIC_FRAME_FRACTION = 0.35   # a spot occupied in >35% of frames is a static false object
STATIONARY_WINDOW = 40         # frames
STATIONARY_MOVE_PX = 45.0      # min movement expected of the real ball over the window


def _remove_static_clutter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ball candidates that belong to a spatially persistent cluster.

    The real ball is almost never at the same spot across many frames, but a
    standing coach / line marking misdetected as a ball is. For each
    candidate, count how many distinct frames have a candidate within
    STATIC_RADIUS_PX; if that exceeds a fraction of all frames, it is static
    clutter and removed -- this is what prevents the tracker locking onto a
    stationary off-ball object.
    """
    if df.empty:
        return df
    xy = df[["cx", "cy"]].to_numpy()
    frames = df["frame"].to_numpy()
    total = max(1, df["frame"].nunique())
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        near = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1]) < STATIC_RADIUS_PX
        if len(np.unique(frames[near])) > STATIC_FRAME_FRACTION * total:
            keep[i] = False
    return df[keep]


def _load_global_candidates(csv: Path | None) -> dict[int, list[tuple[float, float, float]]]:
    pooled: dict[int, list[tuple[float, float, float]]] = {}
    if csv is None:
        return pooled
    df = pd.read_csv(csv)
    df = df[df["class_name"].isin(("sports ball", "frisbee"))].copy()
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = (df["y1"] + df["y2"]) / 2
    df = _remove_static_clutter(df)
    for r in df.itertuples(index=False):
        pooled.setdefault(int(r.frame), []).append((float(r.cx), float(r.cy), float(r.confidence)))
    return pooled


def _detect_in_crop(model, image, cx, cy, window, conf, crop_imgsz):
    """Run the detector on a crop centered at (cx, cy); return ball centers in full-frame coords."""
    h, w = image.shape[:2]
    x1 = max(0, int(cx - window)); x2 = min(w, int(cx + window))
    y1 = max(0, int(cy - window)); y2 = min(h, int(cy + window))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return []
    crop = image[y1:y2, x1:x2]
    result = model.predict(crop, conf=conf, imgsz=crop_imgsz,
                           classes=BALL_COCO_IDS, verbose=False)[0]
    out = []
    for box in result.boxes:
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        out.append((x1 + (bx1 + bx2) / 2, y1 + (by1 + by2) / 2, float(box.conf)))
    return out


def track_ball_search(
    video_path: Path,
    output_csv: Path,
    model_name: str = "yolov8n.pt",
    init_candidates_csv: Path | None = None,
    window: int = 180,
    conf: float = 0.08,
    crop_imgsz: int = 320,
    gate_px: float = GATE_PX,
    max_coast: int = MAX_COAST_FRAMES,
    process_var: float = 25.0,
    measure_var: float = 9.0,
) -> pd.DataFrame:

    model = YOLO(model_name)
    global_cands = _load_global_candidates(init_candidates_csv)

    F, H, Q, R = _kalman_matrices(process_var, measure_var)
    reader = VideoReader(video_path)
    reader.open()

    rows: list[dict] = []
    state: np.ndarray | None = None
    P = np.eye(4) * 1000.0
    coast = 0
    n_window = 0
    recent: list[tuple[float, float]] = []   # recent positions for stationary check

    for frame in tqdm(reader.frames(), total=reader.metadata.frame_count,
                      desc="Ball search", unit="frame"):
        fid = frame.frame_id

        if state is None:                                  # seed from a global candidate
            cands = global_cands.get(fid, [])
            if cands:
                x, y, _ = max(cands, key=lambda c: c[2])
                state = np.array([x, y, 0, 0], dtype=float)
                recent = [(x, y)]
                rows.append({"frame": fid, "x": x, "y": y, "source": "measured"})
            continue

        state = F @ state                                  # predict
        P = F @ P @ F.T + Q
        px, py = state[0], state[1]

        # Hebel 1: detect in the search window around the prediction
        found = _detect_in_crop(model, frame.image, px, py, window, conf, crop_imgsz)
        # fall back to any global candidate in this frame
        found += global_cands.get(fid, [])

        source = "coasted"
        if found:
            x, y, _ = min(found, key=lambda c: np.hypot(c[0] - px, c[1] - py))
            if np.hypot(x - px, y - py) <= gate_px:
                z = np.array([x, y])
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                state = state + K @ (z - H @ state)
                P = (np.eye(4) - K @ H) @ P
                source = "measured"
                coast = 0
                n_window += 1

        if source == "coasted":
            coast += 1
            if coast > max_coast:
                state, P, coast, recent = None, np.eye(4) * 1000.0, 0, []
                continue

        # stationary-rejection safety net: the real ball cannot sit still
        recent.append((float(state[0]), float(state[1])))
        if len(recent) >= STATIONARY_WINDOW:
            recent = recent[-STATIONARY_WINDOW:]
            xs = [p[0] for p in recent]; ys = [p[1] for p in recent]
            span = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if span < STATIONARY_MOVE_PX:                  # locked on a static object
                # drop the wrongly-recorded stationary stretch and re-seed
                rows = [r for r in rows if r["frame"] < fid - STATIONARY_WINDOW]
                state, P, coast, recent = None, np.eye(4) * 1000.0, 0, []
                continue

        rows.append({"frame": fid, "x": float(state[0]), "y": float(state[1]), "source": source})

    reader.close()

    out = pd.DataFrame(rows).sort_values("frame")
    for col in ("x", "y"):                                 # offline smoothing
        s = out[col].rolling(5, center=True, min_periods=1).median()
        out[col] = s.rolling(5, center=True, min_periods=1).mean()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    n_meas = int((out["source"] == "measured").sum())
    total = reader.metadata.frame_count
    print(f"Search-window ball: {len(out)}/{total} frames ({len(out)/total:.0%}) -- "
          f"{n_meas} measured ({n_window} via search window), {len(out)-n_meas} filled")
    return out
