"""
Smooth per-track trajectories to remove jitter -- gap-aware and speed-gated.

Detection boxes jitter a few pixels frame-to-frame, which becomes visible
noise once mapped to pitch coordinates and used for speed/distance metrics.
Two things make naive smoothing wrong on this footage:

  * Tracks have frame *gaps* (a player briefly undetected, especially small
    far ones). A plain rolling window works on row-adjacency, so it happily
    averages the last point before a 90-frame gap with the first point after
    it -- inventing a position that is neither, which then reads as a
    teleport. We instead split each track at real gaps and smooth each
    contiguous segment on its own.
  * In the far half of a side-view camera the homography is compressed, so a
    few pixels of foot-point jitter map to metres. That produces single-frame
    position spikes implying hundreds of km/h. A speed gate drops points whose
    implied velocity is physically impossible before smoothing, and they are
    re-filled by interpolation within the segment.

Smooths pitch_x / pitch_y in place; raw detection boxes are left intact.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 5
DEFAULT_MAX_GAP = 10          # frames; beyond this a track is a new segment
DEFAULT_MAX_SPEED = 12.0      # pitch-units/sec; a sprinter peaks ~11 m/s


def _segment_ids(frames: np.ndarray, max_gap: int) -> np.ndarray:
    """Label contiguous runs, starting a new label wherever a gap exceeds max_gap."""
    breaks = np.diff(frames) > max_gap
    return np.concatenate([[0], np.cumsum(breaks)])


def _gate_speed(x: np.ndarray, y: np.ndarray, frames: np.ndarray,
                max_units_per_frame: np.ndarray) -> tuple[np.ndarray, int]:
    """
    NaN-out points whose step from the previous kept point is impossibly fast.

    Walks the segment forward, holding the last physically-plausible point as
    the anchor; a point too far from it (given the frame delta) is a spike, so
    it is dropped rather than allowed to drag the anchor. Returns (mask_kept,
    n_dropped) -- dropped positions are set to NaN by the caller for interp.
    """
    keep = np.ones(len(x), dtype=bool)
    anchor = 0
    dropped = 0
    for i in range(1, len(x)):
        dframe = frames[i] - frames[anchor]
        step = np.hypot(x[i] - x[anchor], y[i] - y[anchor])
        if step > max_units_per_frame[i] * max(dframe, 1):
            keep[i] = False
            dropped += 1
        else:
            anchor = i
    return keep, dropped


def smooth_tracks_csv(
    input_csv: Path,
    output_csv: Path,
    window: int = DEFAULT_WINDOW,
    max_gap: int = DEFAULT_MAX_GAP,
    fps: float = 25.0,
    max_speed: float = DEFAULT_MAX_SPEED,
    class_name: str = "person",
) -> None:
    """
    Smooth pitch_x / pitch_y per person track: split at frame gaps > ``max_gap``,
    drop points implying speed > ``max_speed`` (units/sec), then apply a
    centered rolling median + mean of size ``window`` within each segment.
    """

    df = pd.read_csv(input_csv)
    if "pitch_x" not in df.columns:
        raise ValueError("Smoothing needs pitch coordinates (run calibration filter first).")

    is_person = (df["class_name"] == class_name) & df["track_id"].notna()
    max_upf = max_speed / fps  # units per frame
    total_dropped = 0

    def _smooth(group: pd.DataFrame) -> pd.DataFrame:
        nonlocal total_dropped
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        seg = _segment_ids(frames, max_gap)
        for col in ("pitch_x", "pitch_y"):
            group[col] = group[col].astype(float)
        for s in np.unique(seg):
            idx = np.where(seg == s)[0]
            x = group["pitch_x"].to_numpy()[idx]
            y = group["pitch_y"].to_numpy()[idx]
            f = frames[idx]
            keep, dropped = _gate_speed(x, y, f, np.full(len(x), max_upf))
            total_dropped += dropped
            xs = pd.Series(np.where(keep, x, np.nan))
            ys = pd.Series(np.where(keep, y, np.nan))
            # re-fill the dropped spikes from their plausible neighbours, then
            # smooth this segment alone (no bleeding across the track's gaps)
            xs = xs.interpolate(limit_direction="both")
            ys = ys.interpolate(limit_direction="both")
            xs = xs.rolling(window, center=True, min_periods=1).median() \
                   .rolling(window, center=True, min_periods=1).mean()
            ys = ys.rolling(window, center=True, min_periods=1).median() \
                   .rolling(window, center=True, min_periods=1).mean()
            gi = group.index[idx]
            group.loc[gi, "pitch_x"] = xs.to_numpy()
            group.loc[gi, "pitch_y"] = ys.to_numpy()
        return group

    persons = df[is_person].groupby("track_id", group_keys=False).apply(_smooth)
    df.loc[is_person, ["pitch_x", "pitch_y"]] = persons[["pitch_x", "pitch_y"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Smoothing: window={window}, gap-split>{max_gap}f, "
          f"speed-gated {total_dropped} spikes (>{max_speed} units/s)")
