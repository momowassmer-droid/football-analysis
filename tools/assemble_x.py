"""Phase 2: assemble the three per-camera ByteTrack runs into one global-id
dataset, and emit the (camera, local id) -> global id mapping the review tool
needs to overlay global ids on the videos.

Views are labeled by camera so the mapping is unambiguous downstream:
    A = match_center, B = match_left, C = match_right.

Inputs : data/output/<clip>/detections/<prefix>teams.csv  (default prefix x_)
Outputs: data/output/review_assembled/<prefix>match_tracks.csv  (global rows)
         data/output/review_assembled/<prefix>idmap.csv         (view,track_id,global_id)

Usage:
    python -m tools.assemble_x            # prefix x_ (ByteTrack)
    python -m tools.assemble_x --prefix xg_
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analytics.match_assembler import assemble_match

ROOT = Path(".")
VIEW_CLIP = {"A": "match_center", "B": "match_left", "C": "match_right"}
OUT_DIR = ROOT / "data" / "output" / "review_assembled"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="x_", help="teams-CSV prefix to assemble")
    ap.add_argument("--match-dist", type=float, default=5.0)
    args = ap.parse_args()

    half_csvs = {}
    for view, clip in VIEW_CLIP.items():
        p = ROOT / "data" / "output" / clip / "detections" / f"{args.prefix}teams.csv"
        if not p.exists():
            print(f"WARN: {p} fehlt -> View {view} übersprungen")
            continue
        half_csvs[view] = p
    if len(half_csvs) < 2:
        print("Zu wenige Views zum Assemblieren.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = assemble_match(
        half_csvs,
        OUT_DIR / f"{args.prefix}match_tracks.csv",
        match_dist=args.match_dist,
        idmap_csv=OUT_DIR / f"{args.prefix}idmap.csv",
    )
    # quick sanity: how many globals seen in >1 camera (the real merges)
    multi = merged[merged["views"].str.len() > 1]["global_id"].nunique()
    print(f"Views: {', '.join(f'{v}={VIEW_CLIP[v]}' for v in half_csvs)}")
    print(f"-> {OUT_DIR}/{args.prefix}match_tracks.csv  (+idmap)")
    print(f"Globale Spieler: {merged['global_id'].nunique()} | in >1 Kamera gesehen: {multi}")


if __name__ == "__main__":
    main()
