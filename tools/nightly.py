"""Nightly runner: rebuild the production tracking, assemble global ids, and
save the result as a timestamped snapshot the review tool can load.

This is the "over-night test" half of the night/day workflow: after a day of
changes, run this once (or let a scheduled task run it) so a dated run appears
in the tool next to the live one -- you compare "last night" vs "today" as A/B
and see at a glance what improved or regressed.

Robust like the auto-labeler: keeps the machine awake for the run, isolates
each pipeline step, prunes old snapshots, and writes an ALL_DONE marker.

Snapshot layout (what the server's ``snap:<ts>:`` runs read):
    data/output/review_runs/<YYYY-MM-DD_HHMM>/
        match_center_teams.csv  match_left_teams.csv  match_right_teams.csv
        idmap.csv               # (view, local id) -> global id, for the global view

Usage:
    python -m tools.nightly            # rebuild + assemble + snapshot
    python -m tools.nightly --snapshot-only   # just snapshot the current output
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output"
RUNS = OUT / "review_runs"
ASSEMBLED = OUT / "review_assembled"
CLIPS = ["match_center", "match_left", "match_right"]
KEEP_SNAPSHOTS = 10

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def _run(module: str, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    print(f"[{time.strftime('%H:%M:%S')}] -> {module} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env, check=True)


def _snapshot() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    dst = RUNS / ts
    dst.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        src = OUT / clip / "detections" / "x_teams.csv"
        if src.exists():
            shutil.copy2(src, dst / f"{clip}_teams.csv")
    idmap = ASSEMBLED / "x_idmap.csv"
    if idmap.exists():
        shutil.copy2(idmap, dst / "idmap.csv")
    return dst


def _prune() -> None:
    snaps = sorted((p for p in RUNS.iterdir() if p.is_dir()), reverse=True)
    for old in snaps[KEEP_SNAPSHOTS:]:
        shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-only", action="store_true",
                    help="skip the rebuild, just snapshot the current output")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    _keep_awake(True)
    start = time.time()
    try:
        if not args.snapshot_only:
            _run("tools.xmodel_pipeline")     # rebuild production tracking (x_)
            _run("tools.assemble_x")          # global ids
        dst = _snapshot()
        _prune()
        (RUNS / "ALL_DONE").write_text(dst.name, encoding="utf-8")
        mins = (time.time() - start) / 60
        print(f"[{time.strftime('%H:%M:%S')}] Snapshot fertig: {dst.name}  ({mins:.1f} min)")
        print("Im Review-Tool als Nacht-Lauf waehlbar (A/B gegen den Live-Lauf).")
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    main()
