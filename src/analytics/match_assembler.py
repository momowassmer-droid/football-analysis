"""
Match assembler: merge several half-field clips into one match dataset.

Each clip is processed independently but calibrated to the *same* pitch
coordinate system (StatsBomb 120 x 80), so all positions live in one space.
Where two views overlap (the middle of the pitch) a player appears in both
at the same pitch position at the same time -- no two players can occupy one
spot -- so tracks that coincide spatially and temporally across views are the
same physical player and are merged onto one global id.

Requires the clips to be frame-synchronised (same time window) and each clip's
detections to already carry pitch_x / pitch_y (+ team_id).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_DISTANCE = 5.0       # pitch units; closer than this across views = same player
MIN_SHARED_FRAMES = 8      # need this many co-present frames to trust a cross-view match


def _trajectories(csv: Path) -> dict[int, pd.DataFrame]:
    df = pd.read_csv(csv)
    df = df[(df["class_name"] == "person") & df["track_id"].notna()].copy()
    df["track_id"] = df["track_id"].astype(int)
    return {tid: g.set_index("frame") for tid, g in df.groupby("track_id")}


def _densify(traj: dict[int, pd.DataFrame]):
    """Stack a view's trajectories into dense (n_tracks, n_frames) arrays so the
    offset search below can use vectorized numpy instead of a Python loop over
    every (offset, track_a, track_b) triple -- the latter is correct but, for
    dozens of tracks times ~180 offsets, means 10^5-10^6 individual pandas
    ``Index.intersection`` calls (each with real overhead), i.e. minutes for a
    tiny clip. Returns (ids, presence_bool, x, y, lo) where column j is frame
    ``lo + j``.
    """
    ids = list(traj)
    if not ids:
        return ids, np.zeros((0, 0), dtype=bool), np.zeros((0, 0)), np.zeros((0, 0)), 0
    lo = min(int(g.index.min()) for g in traj.values())
    hi = max(int(g.index.max()) for g in traj.values())
    n = hi - lo + 1
    pres = np.zeros((len(ids), n), dtype=bool)
    xs = np.zeros((len(ids), n))
    ys = np.zeros((len(ids), n))
    for i, tid in enumerate(ids):
        g = traj[tid]
        fr = g.index.to_numpy() - lo
        pres[i, fr] = True
        xs[i, fr] = g["pitch_x"].to_numpy()
        ys[i, fr] = g["pitch_y"].to_numpy()
    return ids, pres, xs, ys, lo


def estimate_frame_offset(traj_a, traj_b, search_range=90, match_dist=5.0, min_shared=8):
    """
    Find the frame offset that best aligns view B to view A by maximizing the
    number of coincident overlap players (ties broken by smallest distance).
    Returns (best_offset, best_score). A frame f in A corresponds to frame
    f - best_offset in B.

    Vectorized: both views are densified once (see ``_densify``), then every
    offset in the search range is scored with array ops instead of a per-pair
    Python loop -- the same result, orders of magnitude faster.
    """
    _, pres_a, xs_a, ys_a, lo_a = _densify(traj_a)
    _, pres_b, xs_b, ys_b, lo_b = _densify(traj_b)
    n_a, n_b = pres_a.shape[0], pres_b.shape[0]

    best = (0, -1, float("inf"))   # offset, score, total_dist
    if n_a == 0 or n_b == 0:
        return best[0], best[1]

    for offset in range(-search_range, search_range + 1):
        # frame f in A (dense index ia) <-> frame f-offset in B (dense index ia+shift)
        shift = lo_a - lo_b - offset
        ia_lo = max(0, -shift)
        ia_hi = min(pres_a.shape[1], pres_b.shape[1] - shift)
        if ia_hi <= ia_lo:
            continue
        ib_lo = ia_lo + shift

        pa = pres_a[:, ia_lo:ia_hi]
        pb = pres_b[:, ib_lo:ib_lo + (ia_hi - ia_lo)]
        shared = pa[:, None, :] & pb[None, :, :]          # (n_a, n_b, L)
        cnt = shared.sum(axis=2)
        if not np.any(cnt >= min_shared):
            continue

        dx = xs_a[:, ia_lo:ia_hi][:, None, :] - xs_b[:, ib_lo:ib_lo + (ia_hi - ia_lo)][None, :, :]
        dy = ys_a[:, ia_lo:ia_hi][:, None, :] - ys_b[:, ib_lo:ib_lo + (ia_hi - ia_lo)][None, :, :]
        d = np.where(shared, np.hypot(dx, dy), np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(d, axis=2)
        match = (cnt >= min_shared) & (med < match_dist)
        if not np.any(match):
            continue

        score = int(match.sum())
        total_dist = float(med[match].sum())
        if score > best[1] or (score == best[1] and total_dist < best[2]):
            best = (offset, score, total_dist)

    return best[0], best[1]


def _shift_frames(traj: dict, offset: int) -> dict:
    """Relabel a view's frame index by +offset to put it on the reference timeline."""
    return {tid: g.set_index(g.index + offset) for tid, g in traj.items()}


def _cross_view_matches(traj_a, traj_b, match_dist, min_shared):
    """Greedy bipartite matching of A-tracks to B-tracks by spatio-temporal coincidence."""
    candidates = []
    for a_id, a in traj_a.items():
        for b_id, b in traj_b.items():
            shared = a.index.intersection(b.index)
            if len(shared) < min_shared:
                continue
            d = np.hypot(a.loc[shared, "pitch_x"].to_numpy() - b.loc[shared, "pitch_x"].to_numpy(),
                         a.loc[shared, "pitch_y"].to_numpy() - b.loc[shared, "pitch_y"].to_numpy())
            median_d = float(np.median(d))
            if median_d < match_dist:
                candidates.append((median_d, a_id, b_id))

    candidates.sort()
    used_a, used_b, matches = set(), set(), {}
    for _, a_id, b_id in candidates:
        if a_id in used_a or b_id in used_b:
            continue
        used_a.add(a_id); used_b.add(b_id)
        matches[b_id] = a_id          # b_id continues a_id
    return matches


def assemble_match(
    half_csvs: dict[str, Path],
    output_csv: Path,
    match_dist: float = MATCH_DISTANCE,
    min_shared: int = MIN_SHARED_FRAMES,
    auto_sync: bool = True,
    sync_range: int = 90,
    manual_links: list[list] | None = None,
    idmap_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Merge clips in ``half_csvs`` (e.g. {"A": left.csv, "B": right.csv, ...}) into
    one per-frame, global-id dataset. Works for any number of views: tracks are
    matched across *all* view pairs and merged with union-find, so the order of
    the views does not matter and chained coverage (A-B-C) works.

    ``manual_links`` lets the user help where auto-matching fails: each entry is
    a group of [view, track_id] pairs that are the same physical player and are
    force-merged, e.g. [["A", 5], ["B", 12]]. Auto + manual links combine.

    With ``auto_sync`` each view is time-aligned to the best-overlapping earlier
    view, so the clips need not start at the same moment (within +/- sync_range).
    """

    views = list(half_csvs)
    trajs = {v: _trajectories(half_csvs[v]) for v in views}

    if auto_sync and len(views) > 1:
        applied = {views[0]: 0}
        for v in views[1:]:
            best = (0, -1, None)   # offset, score, ref_view
            for ref in applied:
                off, sc = estimate_frame_offset(trajs[ref], trajs[v], sync_range, match_dist, min_shared)
                if sc > best[1]:
                    best = (off + applied[ref], sc, ref)   # cumulative to view 0 timeline
            applied[v] = best[0]
            print(f"Auto-sync view {v}: frame offset {best[0]:+d} vs {best[2]} ({best[1]} players aligned)")
            trajs[v] = _shift_frames(trajs[v], best[0])

    # union-find over every (view, track) node
    parent: dict[tuple[str, int], tuple[str, int]] = {(v, t): (v, t) for v in views for t in trajs[v]}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # auto: match across all view pairs
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            for b_id, a_id in _cross_view_matches(trajs[views[i]], trajs[views[j]], match_dist, min_shared).items():
                union((views[i], a_id), (views[j], b_id))

    # manual: force-merge user-specified groups
    for group in (manual_links or []):
        nodes = [(g[0], int(g[1])) for g in group if (g[0], int(g[1])) in parent]
        for n in nodes[1:]:
            union(nodes[0], n)

    roots = {n: find(n) for n in parent}
    root_to_gid = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    global_id = {n: root_to_gid[roots[n]] for n in parent}

    # build merged per-frame rows
    records: dict[tuple[int, int], list[dict]] = {}
    for v in views:
        for tid, g in trajs[v].items():
            gid = global_id[(v, tid)]
            for frame, row in g.iterrows():
                records.setdefault((int(frame), gid), []).append({
                    "pitch_x": row["pitch_x"], "pitch_y": row["pitch_y"],
                    "team_id": row.get("team_id"), "view": v,
                })

    rows = []
    for (frame, gid), obs in sorted(records.items()):
        teams = [o["team_id"] for o in obs if pd.notna(o["team_id"])]
        rows.append({
            "frame": frame,
            "global_id": gid,
            "track_id": gid,            # alias so existing plotters/analytics work
            "class_name": "person",
            "team_id": int(pd.Series(teams).mode().iloc[0]) if teams else -1,
            "pitch_x": float(np.mean([o["pitch_x"] for o in obs])),
            "pitch_y": float(np.mean([o["pitch_y"] for o in obs])),
            "views": "".join(sorted(o["view"] for o in obs)),
        })

    merged = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    # Optional companion: the (view, local track_id) -> global_id mapping, which
    # the merged table itself does not retain. Needed to relabel per-view image
    # boxes with global ids (e.g. the review tool's cross-camera overlay).
    if idmap_csv is not None:
        idmap_rows = [{"view": v, "track_id": int(t), "global_id": int(global_id[(v, t)])}
                      for (v, t) in parent]
        idmap_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(idmap_rows).sort_values(["view", "track_id"]).to_csv(idmap_csv, index=False)

    n_global = merged["global_id"].nunique()
    n_local = sum(len(trajs[v]) for v in views)
    n_overlap = merged[merged["views"].str.len() > 1]["global_id"].nunique()
    print(f"Match assembler: {n_local} local tracks across {len(views)} views "
          f"-> {n_global} global players ({n_overlap} seen in >1 view)")
    return merged
