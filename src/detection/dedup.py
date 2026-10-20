"""Remove duplicate boxes on the same person without dropping real players.

The x-size football model regularly emits several boxes for one person:
upper-body/feet fragments inside the full-body box, and near-identical
boxes when two classes fire on the same person (goalkeeper+player,
referee+player). Plain IoU-NMS misses the fragments (IoU 0.3-0.5) and a
containment-only rule would hit real players occluding each other, so the
rules below were derived from all 1800 auto-labeled frames of the three
test clips plus visual inspection of sampled pairs (48 crops):

* fragments: small box >=85% inside a bigger box AND <=80% of its height
  -> always the same person in every inspected sample, safe to drop.
* near-identical: IoU >= 0.80 -> physically impossible for two bodies.
* similar-size, IoU 0.5-0.8: duplicates and genuinely close players are
  inseparable per frame (measured: both populate IoU 0.61-0.69). These are
  resolved at TRACK-PAIR level: when two tracks share >= 10 frames whose
  boxes sit inside each other (IoMin >= 0.75), that pair is one person
  double-detected in episodes -- real players only cross for a handful of
  frames. On each glued frame the smaller box is dropped, regardless of
  which track id carries it: the match_right keeper showed that the greedy
  tracker weaves fragment and full-body boxes into BOTH ids over time, so
  dropping a whole "duplicate track" would drop real coverage. Containment,
  not IoU, is the glue metric (an upper-body fragment has IoU only ~0.3
  against its full-body parent but containment ~0.85). An appearance veto
  (caller-side, see tools/xmodel_pipeline.py) adds a second safety net.

Frame-level entry point: :func:`dedup_frame_boxes`.
Track-pair entry point: :func:`glued_pair_frame_drops`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# x-model class ids: 0=ball, 1=goalkeeper, 2=player, 3=referee
PERSON_CLASSES = {1, 2, 3}
# goalkeeper/referee are more specific than player: when a near-identical
# pair mixes classes, the surviving box adopts the specific class.
_SPECIFIC = {1, 3}

FRAGMENT_IOMIN = 0.85
FRAGMENT_MAX_H_RATIO = 0.80
# Second band for a clear body-part fragment (head/feet) that pokes slightly
# outside the full-body box, so its containment is lower -- but it is much
# shorter, which no genuine second player standing behind another shows at this
# containment. Verified on 210 sampled survivors (12/12 were head/foot fragments
# of the SAME person, none a far player behind a near one).
FRAGMENT2_IOMIN = 0.70
FRAGMENT2_MAX_H_RATIO = 0.60
IDENTICAL_IOU = 0.80


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def _inter(a: Box, b: Box) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    return max(0.0, iw) * max(0.0, ih)


def iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    return i / (a.area + b.area - i) if i > 0 else 0.0


def iomin(a: Box, b: Box) -> float:
    i = _inter(a, b)
    m = min(a.area, b.area)
    return i / m if m > 0 else 0.0


def dedup_frame_boxes(boxes: list[Box]) -> tuple[list[Box], list[tuple[Box, Box, str]]]:
    """Frame-level dedup (stage 1). Returns (kept, removed).

    ``removed`` entries are (removed_box, kept_partner_box, rule) so callers
    can log/render proof that every removal had a surviving box on the same
    person. Non-person classes (ball) pass through untouched.
    """
    persons = sorted((b for b in boxes if b.cls in PERSON_CLASSES),
                     key=lambda b: b.area, reverse=True)
    other = [b for b in boxes if b.cls not in PERSON_CLASSES]
    kept: list[Box] = []
    removed: list[tuple[Box, Box, str]] = []
    for box in persons:
        partner = None
        rule = ""
        for k in kept:  # kept boxes are all >= box.area
            im = iomin(box, k)
            hr = box.height / k.height
            if (im >= FRAGMENT_IOMIN and hr <= FRAGMENT_MAX_H_RATIO) or \
               (im >= FRAGMENT2_IOMIN and hr <= FRAGMENT2_MAX_H_RATIO):
                partner, rule = k, "fragment"
                break
            if iou(box, k) >= IDENTICAL_IOU:
                partner, rule = k, "identical"
                break
        if partner is None:
            kept.append(box)
        else:
            if rule == "identical" and box.cls in _SPECIFIC and partner.cls not in _SPECIFIC:
                partner.cls = box.cls
            removed.append((box, partner, rule))
    return kept + other, removed


def glued_pair_frame_drops(
    df: pd.DataFrame,
    min_glued_frames: int = 10,
    iomin_threshold: float = 0.75,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Track-pair dedup (stage 2) on a detection table.

    ``df`` needs columns frame, track_id, x1, y1, x2, y2. A pair of tracks
    qualifies as "glued" (same person double-detected in episodes) when at
    least ``min_glued_frames`` shared frames have box containment
    (IoMin) >= ``iomin_threshold``. Real players crossing each other stay
    below that for lack of frames; the threshold was chosen from measured
    episode lengths (crossings: 1-13 frames, median 1).

    On every glued frame the SMALLER box of that frame is marked for
    removal -- identity-agnostic, because the tracker may weave the
    fragment through both track ids. The bigger box always survives, so
    the person keeps coverage on every frame.

    Returns {(track_a, track_b): (drop_list, median_h_ratio)} with
    drop_list = [(frame, track_id_to_drop), ...] and median_h_ratio the
    median small/big box-height ratio over the glued frames. Callers
    should apply an appearance veto ONLY to similar-size pairs
    (h_ratio > ~0.8): fragment crops never align region-wise with their
    full-body parent, so appearance comparison is meaningless there.
    """
    tracks: dict[int, pd.DataFrame] = {
        int(t): g.set_index("frame")[["x1", "y1", "x2", "y2"]].sort_index()
        for t, g in df.groupby("track_id")
    }
    ids = sorted(tracks)
    drops: dict[tuple[int, int], tuple[list[tuple[int, int]], float]] = {}
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            ta, tb = tracks[a], tracks[b]
            shared = ta.index.intersection(tb.index)
            if len(shared) < min_glued_frames:
                continue
            va = ta.loc[shared].to_numpy()
            vb = tb.loc[shared].to_numpy()
            iw = np.minimum(va[:, 2], vb[:, 2]) - np.maximum(va[:, 0], vb[:, 0])
            ih = np.minimum(va[:, 3], vb[:, 3]) - np.maximum(va[:, 1], vb[:, 1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            area_a = (va[:, 2] - va[:, 0]) * (va[:, 3] - va[:, 1])
            area_b = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
            iomins = inter / np.maximum(np.minimum(area_a, area_b), 1e-9)
            glued = iomins >= iomin_threshold
            if int(glued.sum()) < min_glued_frames:
                continue
            pair_drops = [
                (int(f), a if area_a[k] < area_b[k] else b)
                for k, f in enumerate(shared) if glued[k]
            ]
            ha = va[:, 3] - va[:, 1]
            hb = vb[:, 3] - vb[:, 1]
            h_ratio = float(np.median(
                (np.minimum(ha, hb) / np.maximum(ha, hb))[glued]))
            drops[(a, b)] = (pair_drops, h_ratio)
    return drops
