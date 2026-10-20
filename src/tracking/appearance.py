"""
Appearance fingerprints for player re-identification.

The enhanced stitcher originally gated fragment merges on a single median
torso colour per track -- three numbers. That cannot tell two same-coloured
players apart (both white shirts collapse to the same point), which both
misses correct re-links and allows wrong ones. This module replaces that with
a richer, still-cheap fingerprint:

  * grass-masked Hue-Saturation histogram (brightness-robust, so a player in
    sun vs shadow still matches), computed separately for the upper body
    (shirt) and lower body (shorts/socks) -- the two together are far more
    discriminative than one blended colour,
  * sampled only every ``stride`` frames, so it costs a *fraction* of the old
    per-frame colour pass rather than more,
  * aggregated per track by averaging the per-crop histograms, then compared
    with the Bhattacharyya distance (0 = identical, 1 = disjoint).

Reusable beyond stitching: the same per-track fingerprint powers mid-clip
ID-swap detection (a track whose fingerprint suddenly, persistently changes
has picked up a different player).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.video.video_reader import VideoReader

# Torso/legs sampling: the box's central column, split top/bottom. Avoids the
# arms and the ground sliver at the very bottom of the box.
_X_RANGE = (0.25, 0.75)
_UPPER_Y = (0.15, 0.55)   # shirt
_LOWER_Y = (0.55, 0.90)   # shorts / socks

_H_BINS, _S_BINS = 12, 8   # HS histogram resolution (per body region)
_MIN_BOX_PX = 24           # skip boxes too small for a meaningful histogram
_MIN_JERSEY_FRAC = 0.15    # crop must be at least this non-grass to be used

# grass hue/sat band (shared convention with the team classifier / masks)
_GRASS_HUE = (35, 90)
_GRASS_SAT_MIN = 60


def _region_hist(region_bgr: np.ndarray) -> np.ndarray | None:
    """Grass-masked, normalised HS histogram of one body-region crop."""
    if region_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    grass = (hue >= _GRASS_HUE[0]) & (hue <= _GRASS_HUE[1]) & (sat > _GRASS_SAT_MIN)
    mask = (~grass).astype(np.uint8)
    if int(mask.sum()) < max(8, _MIN_JERSEY_FRAC * mask.size):
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [_H_BINS, _S_BINS], [0, 180, 0, 256])
    total = hist.sum()
    if total <= 0:
        return None
    return (hist / total).flatten().astype(np.float32)


def _crop_fingerprint(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray | None:
    """Concatenated upper+lower HS histograms for one detection box."""
    w, h = x2 - x1, y2 - y1
    if w < _MIN_BOX_PX or h < _MIN_BOX_PX:
        return None
    cx1, cx2 = int(x1 + w * _X_RANGE[0]), int(x1 + w * _X_RANGE[1])

    def region(y_lo: float, y_hi: float) -> np.ndarray:
        ry1, ry2 = int(y1 + h * y_lo), int(y1 + h * y_hi)
        return frame[max(ry1, 0):max(ry2, 1), max(cx1, 0):max(cx2, 1)]

    upper = _region_hist(region(*_UPPER_Y))
    lower = _region_hist(region(*_LOWER_Y))
    if upper is None and lower is None:
        return None
    # Missing half -> zeros, so the present half still carries identity. The
    # halves are kept separate (not summed) so shirt and shorts stay distinct.
    zeros = np.zeros(_H_BINS * _S_BINS, dtype=np.float32)
    return np.concatenate([upper if upper is not None else zeros,
                           lower if lower is not None else zeros])


def extract_track_appearance(
    video_path: Path,
    df: pd.DataFrame,
    stride: int = 5,
    class_name: str = "person",
) -> dict[int, np.ndarray]:
    """
    Per-track appearance fingerprint, sampling one frame in every ``stride``.

    Returns ``{track_id: fingerprint}`` where the fingerprint is the mean of
    that track's per-crop upper+lower HS histograms. Tracks with no usable
    (large-enough, non-grass) crop are absent -- the caller treats that as
    "unknown", not as a match.
    """
    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"])
    by_frame = {fid: g for fid, g in persons.groupby("frame")}

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    reader = VideoReader(video_path)
    reader.open()
    for frame in reader.frames():
        if frame.frame_id % stride != 0:          # only every N-th frame
            continue
        rows = by_frame.get(frame.frame_id)
        if rows is None:
            continue
        for _, row in rows.iterrows():
            fp = _crop_fingerprint(frame.image, row["x1"], row["y1"], row["x2"], row["y2"])
            if fp is None:
                continue
            tid = int(row["track_id"])
            if tid in sums:
                sums[tid] += fp
                counts[tid] += 1
            else:
                sums[tid] = fp.copy()
                counts[tid] = 1
    reader.close()

    return {tid: sums[tid] / counts[tid] for tid in sums}


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """
    Bhattacharyya distance between two fingerprints (0 identical .. 1 disjoint).

    Upper and lower halves are compared separately and averaged, so a matching
    shirt is not washed out by differing shorts (or vice versa). ``None`` when
    either fingerprint is missing -- the caller decides how to treat unknowns.
    """
    if a is None or b is None:
        return None
    half = _H_BINS * _S_BINS
    d_upper = cv2.compareHist(a[:half], b[:half], cv2.HISTCMP_BHATTACHARYYA)
    d_lower = cv2.compareHist(a[half:], b[half:], cv2.HISTCMP_BHATTACHARYYA)
    return float((d_upper + d_lower) / 2.0)
