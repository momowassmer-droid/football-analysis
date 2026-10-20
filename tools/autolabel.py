"""
Robust, resumable auto-labeler: run a strong (slow) model over full clips to
produce YOLO-format training data for a smaller model.

Built to run unattended overnight and *never lose progress*:

  * Resumable -- a progress file records the last finished frame; re-running the
    exact same command continues from there. A crash, reboot, or Ctrl-C costs
    at most one frame, never the whole run.
  * Per-frame isolation -- one unreadable/odd frame is logged and skipped, it
    does not kill the run.
  * Incremental writes -- every frame's image + label is written and flushed
    immediately, so results already on disk survive any later failure.
  * Keeps the machine awake -- Windows is told not to sleep/hibernate while the
    script runs (no admin needed), the single most common cause of an
    "overnight job that silently stopped".
  * Heartbeat log -- progress + ETA appended to a log file so you can check
    from another window without disturbing it.

Usage (one clip):
    python -m tools.autolabel VIDEO OUTDIR [--model M] [--stride N] [--conf C]

Usage (all three test clips in sequence):
    python -m tools.autolabel --batch
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2

# --- keep Windows awake for the whole run (no admin required) ---------------
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes
    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def _log(logfile: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with logfile.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_progress(progress_file: Path, frame_id: int) -> None:
    """Atomic update: write to a temp file then replace, so a crash mid-write
    can never corrupt the progress marker."""
    tmp = progress_file.with_suffix(".tmp")
    tmp.write_text(str(frame_id), encoding="utf-8")
    tmp.replace(progress_file)


def _read_progress(progress_file: Path) -> int:
    if progress_file.exists():
        try:
            return int(progress_file.read_text(encoding="utf-8").strip())
        except ValueError:
            return -1
    return -1


def autolabel_video(
    video: Path,
    outdir: Path,
    model_path: str = "models/football_x_ref.pt",
    stride: int = 1,
    conf: float = 0.25,
    imgsz: int = 1920,
) -> None:
    from ultralytics import YOLO

    images = outdir / "images"
    labels = outdir / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    progress_file = outdir / "progress.txt"
    logfile = outdir / "autolabel.log"

    done_upto = _read_progress(progress_file)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _log(logfile, f"=== {video.name}: {total} frames, resume after frame {done_upto} ===")
    model = YOLO(model_path)

    start = time.time()
    processed = 0
    fid = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid <= done_upto or (fid % stride != 0):
            continue
        try:
            res = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            lines = []
            for box in res.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stem = f"{video.stem}_{fid:06d}"
            cv2.imwrite(str(images / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            (labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            _write_progress(progress_file, fid)     # only after image+label are on disk
            processed += 1
            if processed % 25 == 0:
                rate = (time.time() - start) / processed
                eta = rate * (total - fid) / 60
                _log(logfile, f"frame {fid}/{total}  {rate:.1f}s/frame  ~{eta:.0f} min left")
        except Exception as exc:                      # one bad frame must not stop the night
            _log(logfile, f"frame {fid}: SKIP ({exc})\n{traceback.format_exc()}")
    cap.release()
    _log(logfile, f"=== {video.name}: DONE, {processed} new frames labelled ===")


_BATCH = [
    ("data/raw/test1/match_center.mp4", "data/autolabel/match_center"),
    ("data/raw/test1/match_left.mp4",   "data/autolabel/match_left"),
    ("data/raw/test1/match_right.mp4",  "data/autolabel/match_right"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--batch", action="store_true", help="process all three test clips")
    ap.add_argument("--model", default="models/football_x_ref.pt")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    _keep_awake(True)
    try:
        if args.batch:
            for v, o in _BATCH:
                autolabel_video(Path(v), Path(o), args.model, args.stride, args.conf)
            # Completion marker: the restart wrapper stops looping once this exists.
            Path("data/autolabel/ALL_DONE").write_text(
                time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        elif args.video and args.outdir:
            autolabel_video(Path(args.video), Path(args.outdir), args.model,
                            args.stride, args.conf)
        else:
            ap.print_help()
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    main()
