"""Local review tool for football tracking runs.

A dependency-free (stdlib only) HTTP server that powers a browser app to
watch the three camera clips in sync with their tracking overlays, follow
individual track IDs, compare two runs A/B, and record per-track feedback.

Why stdlib and not Flask: Flask is not installed and the tool must run on the
offline CPU laptop without adding dependencies. ``http.server`` is enough --
the only non-trivial bit is HTTP Range support so the browser can seek inside
the mp4s (``<video>`` scrubbing issues Range requests).

A "run" is just a filename prefix inside ``data/output/<clip>/detections/``:
e.g. ``x_`` -> ``x_teams.csv`` (current ByteTrack result). Any prefix that has
a ``*teams.csv`` for at least one clip shows up as a selectable run, so A/B is
simply picking two prefixes.

Usage:
    python -m tools.review.server [--port 8731]
then open http://127.0.0.1:8731/ .
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "test1"
OUT = ROOT / "data" / "output"
FEEDBACK_DIR = OUT / "review_feedback"
ASSEMBLED = OUT / "review_assembled"

CLIPS = ["match_center", "match_left", "match_right"]
FPS = 30.0

# friendly labels for known prefixes; unknown prefixes fall back to the prefix
RUN_LABELS = {"": "nano (alt)", "x_": "ByteTrack", "xg_": "Greedy",
              "xbt_": "ByteTrack", "xs_": "ByteTrack+Stitch"}

# camera <-> assembler view letter (must match tools/assemble_x.py)
VIEW_CLIP = {"A": "match_center", "B": "match_left", "C": "match_right"}
CLIP_VIEW = {c: v for v, c in VIEW_CLIP.items()}
GLOBAL_SUFFIX = "@global"
RUNS_DIR = OUT / "review_runs"      # timestamped snapshots from the nightly runner


def _detections_dir(clip: str) -> Path:
    return OUT / clip / "detections"


def _run_paths(prefix: str, clip: str) -> tuple[Path, Path, bool]:
    """Resolve a run prefix to (teams_csv, idmap_csv, is_global).

    A live run is a filename prefix in the per-clip detections dir. A snapshot
    run is ``snap:<timestamp>:`` and reads from ``review_runs/<timestamp>/``.
    ``@global`` on either uses the idmap to relabel local ids to global ids.
    """
    is_global = prefix.endswith(GLOBAL_SUFFIX)
    base = prefix[:-len(GLOBAL_SUFFIX)] if is_global else prefix
    if base.startswith("snap:"):
        ts = base[len("snap:"):].rstrip(":")
        return RUNS_DIR / ts / f"{clip}_teams.csv", RUNS_DIR / ts / "idmap.csv", is_global
    return _detections_dir(clip) / f"{base}teams.csv", ASSEMBLED / f"{base}idmap.csv", is_global


def discover_runs() -> list[dict]:
    """Every teams-CSV prefix that exists for at least one clip becomes a run."""
    prefixes: dict[str, list[str]] = {}
    for clip in CLIPS:
        d = _detections_dir(clip)
        if not d.exists():
            continue
        for f in d.glob("*teams.csv"):
            m = re.match(r"(.*)teams\.csv$", f.name)
            prefix = m.group(1) if m else ""
            prefixes.setdefault(prefix, [])
            if clip not in prefixes[prefix]:
                prefixes[prefix].append(clip)
    runs = []
    for prefix, clips in sorted(prefixes.items()):
        # The nano run (prefix "") tracks the crowd and is not our production
        # model; hide it so spectators never leak into what the tool presents.
        if prefix == "":
            continue
        runs.append({
            "prefix": prefix,
            "label": RUN_LABELS.get(prefix, prefix.rstrip("_") or "?"),
            "clips": clips,
            "global": False,
        })
    # add a global (assembled) run for every base prefix that has an idmap
    for idmap in sorted(ASSEMBLED.glob("*idmap.csv")):
        base = idmap.name[:-len("idmap.csv")]
        runs.append({
            "prefix": base + GLOBAL_SUFFIX,
            "label": f"{RUN_LABELS.get(base, base.rstrip('_'))} · global",
            "clips": CLIPS,
            "global": True,
        })
    # timestamped snapshots from the nightly runner (newest first), so the tool
    # can compare "last night" vs "today"
    if RUNS_DIR.exists():
        for d in sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True):
            clips = [c for c in CLIPS if (d / f"{c}_teams.csv").exists()]
            if not clips:
                continue
            runs.append({"prefix": f"snap:{d.name}:", "label": f"🌙 {d.name}",
                         "clips": clips, "global": False})
            if (d / "idmap.csv").exists():
                runs.append({"prefix": f"snap:{d.name}:{GLOBAL_SUFFIX}",
                             "label": f"🌙 {d.name} · global", "clips": CLIPS, "global": True})
    return runs


@lru_cache(maxsize=16)
def _idmap(idmap_path: str) -> tuple:
    """(clip, local_id) -> global_id for an assembled run. Returned as a tuple
    of items so it is hashable/cacheable; callers rebuild the dict."""
    path = Path(idmap_path)
    if not path.exists():
        return ()
    df = pd.read_csv(path)
    out = {}
    for r in df.itertuples(index=False):
        clip = VIEW_CLIP.get(str(r.view))
        if clip is not None:
            out[(clip, int(r.track_id))] = int(r.global_id)
    return tuple(out.items())


@lru_cache(maxsize=64)
def load_tracks(prefix: str, clip: str) -> str:
    """Compact JSON payload for one run+clip: per-box rows + pitch coords.

    Returned as a JSON string (cached) so repeated requests are cheap. Rows are
    arrays to keep the payload small: [frame, id, x1, y1, x2, y2, team, px, py].
    ``px/py`` are StatsBomb pitch coords (may be null when off-pitch).

    For a global run (prefix ends with ``@global``) the box ``id`` is the
    cross-camera global id: boxes come from the base run's per-camera CSV but
    their local track_id is remapped via the assembler's idmap, so the same
    player carries one id across all three cameras.
    """
    csv, idmap_path, is_global = _run_paths(prefix, clip)
    idmap = dict(_idmap(str(idmap_path))) if is_global else None
    if not csv.exists():
        return json.dumps({"fps": FPS, "boxes": [], "ids": []})
    df = pd.read_csv(csv)
    df = df[df["class_name"] == "person"] if "class_name" in df.columns else df
    for col in ("pitch_x", "pitch_y", "team_id"):
        if col not in df.columns:
            df[col] = None
    rows = []
    for r in df.itertuples(index=False):
        local_id = int(r.track_id)
        box_id = local_id
        if is_global:
            gid = idmap.get((clip, local_id))
            if gid is None:
                continue          # unassembled track (shouldn't happen: every track has a gid)
            box_id = gid
        px = getattr(r, "pitch_x")
        py = getattr(r, "pitch_y")
        team = getattr(r, "team_id")
        rows.append([
            int(r.frame), box_id,
            round(float(r.x1), 1), round(float(r.y1), 1),
            round(float(r.x2), 1), round(float(r.y2), 1),
            (None if pd.isna(team) else int(team)),
            (None if pd.isna(px) else round(float(px), 2)),
            (None if pd.isna(py) else round(float(py), 2)),
        ])
    ids = sorted({row[1] for row in rows})
    return json.dumps({"fps": FPS, "boxes": rows, "ids": ids})


class Handler(BaseHTTPRequestHandler):
    server_version = "ReviewTool/0.1"

    def log_message(self, *args):  # quieter console
        pass

    # ---- helpers ---------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = obj if isinstance(obj, str) else json.dumps(obj)
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_video(self, path: Path):
        """Serve an mp4 honoring a single Range header so <video> can seek."""
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    # ---- routing ---------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query)
        if path in ("/", "/index.html"):
            return self._send_static(HERE / "index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send_static(HERE / "app.js", "application/javascript; charset=utf-8")
        if path == "/styles.css":
            return self._send_static(HERE / "styles.css", "text/css; charset=utf-8")
        if path == "/api/runs":
            return self._send_json({"runs": discover_runs(), "clips": CLIPS, "fps": FPS})
        if path == "/api/tracks":
            prefix = q.get("run", ["x_"])[0]
            clip = q.get("clip", ["match_center"])[0]
            if clip not in CLIPS:
                return self._send_json({"error": "unknown clip"}, 400)
            return self._send_json(load_tracks(prefix, clip))
        if path.startswith("/video/"):
            name = path[len("/video/"):]
            clip = name[:-4] if name.endswith(".mp4") else name
            if clip not in CLIPS:
                return self._send_json({"error": "unknown clip"}, 404)
            vp = RAW / f"{clip}.mp4"
            if not vp.exists():
                return self._send_json({"error": "video missing"}, 404)
            return self._send_video(vp)
        if path == "/api/feedback":
            return self._send_json({"feedback": _read_feedback(q.get("run", ["x_"])[0])})
        return self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/feedback":
            return self._send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        run = payload.get("run", "x_")
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "clip": payload.get("clip"),
            "track_id": payload.get("track_id"),
            "issue": payload.get("issue"),
            "frame": payload.get("frame"),
            "note": payload.get("note", ""),
        }
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        fp = FEEDBACK_DIR / f"{run.rstrip('_') or 'run'}.json"
        existing = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
        existing.append(entry)
        fp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        return self._send_json({"ok": True, "count": len(existing)})


def _read_feedback(run: str) -> list:
    fp = FEEDBACK_DIR / f"{run.rstrip('_') or 'run'}.json"
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review-Tool laeuft auf http://127.0.0.1:{args.port}/  (Strg+C zum Beenden)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
