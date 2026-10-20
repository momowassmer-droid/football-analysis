"""
Kalman ball tracker with multi-model ensemble input.

The ball is the pipeline's weakest signal because the generic detector
sees it in only a fraction of frames. Two ideas maximize coverage without
a dedicated model:

  1. Ensemble: different YOLO generations detect the ball in *different*
     frames, so the union of their ball candidates raises recall.
  2. Single-ball physics: a constant-velocity Kalman filter picks, among a
     frame's candidates, the one closest to its prediction (gating away
     false positives like the "frisbee" misclassification), and fills the
     remaining gaps by prediction. A backward RTS-style pass then smooths
     the whole trajectory.

Runs in pixel space (seconds on CPU); map to pitch afterwards if needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BALL_CLASSES = ("sports ball", "frisbee")
GATE_PX = 120.0
MAX_COAST_FRAMES = 20


def _candidates(detection_csvs: list[Path], ball_classes: tuple[str, ...]) -> dict[int, list[tuple[float, float, float]]]:
    """frame -> list of (x, y, confidence) ball candidates, pooled over all models."""
    pooled: dict[int, list[tuple[float, float, float]]] = {}
    for csv in detection_csvs:
        df = pd.read_csv(csv)
        cand = df[df["class_name"].isin(ball_classes)]
        for r in cand.itertuples(index=False):
            x = (r.x1 + r.x2) / 2
            y = (r.y1 + r.y2) / 2
            pooled.setdefault(int(r.frame), []).append((float(x), float(y), float(r.confidence)))
    return pooled


def _kalman_matrices(process_var: float, measure_var: float):
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Q = np.eye(4) * process_var
    R = np.eye(2) * measure_var
    return F, H, Q, R


def track_ball_kalman(
    detection_csvs: list[Path] | Path,
    output_csv: Path,
    ball_classes: tuple[str, ...] = DEFAULT_BALL_CLASSES,
    gate_px: float = GATE_PX,
    max_coast: int = MAX_COAST_FRAMES,
    process_var: float = 25.0,
    measure_var: float = 9.0,
) -> pd.DataFrame:

    if isinstance(detection_csvs, Path):
        detection_csvs = [detection_csvs]

    pooled = _candidates(detection_csvs, ball_classes)
    if not pooled:
        print("Kalman ball: no ball candidates found")
        pd.DataFrame(columns=["frame", "x", "y", "source"]).to_csv(output_csv, index=False)
        return pd.DataFrame()

    F, H, Q, R = _kalman_matrices(process_var, measure_var)
    first, last = min(pooled), max(pooled)

    rows: list[dict] = []
    state: np.ndarray | None = None
    P = np.eye(4) * 1000.0
    coast = 0

    for frame in range(first, last + 1):
        cands = pooled.get(frame, [])

        if state is None:
            if cands:                                       # init on most confident candidate
                x, y, _ = max(cands, key=lambda c: c[2])
                state = np.array([x, y, 0, 0], dtype=float)
                rows.append({"frame": frame, "x": x, "y": y, "source": "measured"})
            continue

        state = F @ state                                   # predict
        P = F @ P @ F.T + Q

        source = "coasted"
        if cands:
            px, py = state[0], state[1]
            # choose the candidate nearest the prediction (gates out false positives)
            x, y, _ = min(cands, key=lambda c: np.hypot(c[0] - px, c[1] - py))
            if np.hypot(x - px, y - py) <= gate_px:
                z = np.array([x, y])
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                state = state + K @ (z - H @ state)
                P = (np.eye(4) - K @ H) @ P
                source = "measured"
                coast = 0

        if source == "coasted":
            coast += 1
            if coast > max_coast:
                state, P, coast = None, np.eye(4) * 1000.0, 0
                continue

        rows.append({"frame": frame, "x": float(state[0]), "y": float(state[1]), "source": source})

    out = pd.DataFrame(rows)
    _smooth_inplace(out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    n_meas = int((out["source"] == "measured").sum())
    total = last - first + 1
    print(f"Kalman ball: {len(out)}/{total} frames ({len(out)/total:.0%}) "
          f"-- {n_meas} measured (ensemble of {len(detection_csvs)} models), {len(out)-n_meas} filled")
    return out


def _smooth_inplace(out: pd.DataFrame, window: int = 5) -> None:
    """Centered rolling-median+mean smoothing of the trajectory (offline pass)."""
    if out.empty:
        return
    out.sort_values("frame", inplace=True)
    for col in ("x", "y"):
        s = out[col].rolling(window, center=True, min_periods=1).median()
        out[col] = s.rolling(window, center=True, min_periods=1).mean()
