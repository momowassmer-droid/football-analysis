"""Split a track where its appearance changes abruptly -- i.e. where the same
track id was handed to a different-looking player after a crossing.

ByteTrack here associates on motion/IoU only (the x-model boxes carry no
appearance), so at a crossing an id can follow the wrong player. This pass
looks, per track, for a single frame where the fingerprint of everything
before it differs strongly and persistently from everything after it, and
cuts the track there into two ids.

Deliberately conservative: it only cuts on a strong, *sustained* change
(both sides need several samples), so a mere lighting/pose wobble of one
player does not trigger a split. Same-kit swaps stay invisible to appearance
and are not caught -- that is an honest limit of CPU tracking without re-id.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.tracking.appearance import _crop_fingerprint, appearance_distance

SPLIT_STRIDE = 5      # sample one frame in five for fingerprints
MIN_SEGMENT = 4       # samples required on each side of a cut (>=20 frames each)
SPLIT_THRESHOLD = 0.6  # Bhattacharyya distance; > this = clearly different player


def _per_sample_fingerprints(video: Path, df: pd.DataFrame, stride: int) -> dict[int, list]:
    sampled = df[df["frame"] % stride == 0]
    by_frame = {int(f): g for f, g in sampled.groupby("frame")}
    out: dict[int, list] = defaultdict(list)
    cap = cv2.VideoCapture(str(video))
    for f in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        for r in by_frame[f].itertuples(index=False):
            fp = _crop_fingerprint(img, r.x1, r.y1, r.x2, r.y2)
            if fp is not None:
                out[int(r.track_id)].append((int(r.frame), fp))
    cap.release()
    return out


def _best_split(samples: list, min_seg: int, thresh: float) -> int | None:
    """Frame at which to cut, or None. Cut = argmax over interior points of the
    Bhattacharyya distance between the mean fingerprint before and after."""
    n = len(samples)
    if n < 2 * min_seg:
        return None
    fps = [s[1] for s in samples]
    best_i, best_d = None, 0.0
    for i in range(min_seg, n - min_seg + 1):
        a = np.mean(fps[:i], axis=0).astype(np.float32)
        b = np.mean(fps[i:], axis=0).astype(np.float32)
        d = appearance_distance(a, b)
        if d is not None and d > best_d:
            best_i, best_d = i, d
    if best_i is not None and best_d >= thresh:
        return samples[best_i][0]
    return None


def split_id_switches_csv(
    video_path: Path,
    input_csv: Path,
    output_csv: Path,
    stride: int = SPLIT_STRIDE,
    min_segment: int = MIN_SEGMENT,
    threshold: float = SPLIT_THRESHOLD,
    class_name: str = "person",
) -> int:
    """Cut appearance-swapped tracks. Returns the number of tracks split."""
    df = pd.read_csv(input_csv)
    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"]).copy()
    persons["track_id"] = persons["track_id"].astype(int)
    samples = _per_sample_fingerprints(video_path, persons, stride)

    next_id = int(df["track_id"].max()) + 1 if df["track_id"].notna().any() else 1
    n_split = 0
    for tid, s in samples.items():
        cut = _best_split(s, min_segment, threshold)
        if cut is None:
            continue
        mask = (df["track_id"] == tid) & (df["frame"] >= cut)
        df.loc[mask, "track_id"] = next_id
        next_id += 1
        n_split += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"ID-Switch-Split: {n_split} Tracks geteilt (Aussehenswechsel >= {threshold})")
    return n_split
