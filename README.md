# Football Analysis — Multi-Camera Player Tracking

Turn amateur football footage (Veo multi-camera) into calibrated, StatsBomb-format
player-tracking data. An end-to-end computer-vision pipeline built from scratch:
camera-motion estimation, pitch calibration, detection, multi-object tracking,
multi-camera fusion, and an interactive review tool.

> Personal project · CPU-only (GPU planned) · currently validated on short test clips.

## Pipeline

```
video → camera motion → pitch calibration → detection → tracking → cleanup
      → teams → multi-camera fusion (global IDs) → analytics (run paths, heatmaps, …)
```

## Highlights

- **Camera-motion module** — per-frame homography onto a reference frame (ORB features
  on a grass mask, direct-to-reference matching, temporal smoothing, a photometric
  verifier that checks the chosen transform per frame).
- **Pitch calibration** — image↔pitch homography (least-squares) from the real venue
  dimensions, snap-to-line/circle, a leave-one-out advisor for the least consistent point.
- **Detection** — a football-specific YOLOv8x model used as an offline auto-labeler;
  two-stage deduplication (body-part fragments, glued track pairs), ball/artifact removal
  by box shape, and an off-field filter for spectators/staff.
- **Tracking** — ByteTrack driven directly on the pre-computed boxes (milliseconds instead
  of 9 s/frame), with association done in the **stabilized reference frame** so camera pan
  no longer breaks it; Kalman filter + Hungarian assignment.
- **Multi-camera fusion** — union-find merge of spatio-temporally coincident tracks across
  the three views into **global player IDs**, with automatic frame-offset sync.
- **Interactive review tool** (`tools/review/`) — a local browser app backed by a Python
  **standard-library** server (HTTP range for video scrubbing, no framework): three synced
  videos with box/ID overlays, per-camera **and** global ID filtering, A/B comparison of two
  runs, a coupled run-path pitch map, per-track feedback capture, and dated nightly snapshots.
- **Analytics** — run paths, team heatmaps, player distances (real metres), team compactness
  and formation, all in StatsBomb pitch coordinates (120×80).

## Engineering approach

- **Visual proof over metrics** — every change validated against rendered images; several
  times a metric (track count, brightness score) was misleading and only the visual told the
  truth.
- **Measure before fixing** — problems characterized on the real data before any threshold
  was changed.
- **Honest negative results** — approaches that regressed on this material were rejected
  rather than shipped; limitations are named explicitly.
- **Living documentation** — see [`ARCHITECTURE.md`](ARCHITECTURE.md) (module-by-module
  breakdown) and [`PLAN.md`](PLAN.md) (roadmap + changelog).

## Tech stack

Python · OpenCV · ultralytics YOLO (YOLOv8x) · ByteTrack · NumPy / Pandas ·
projective geometry / homography · mplsoccer · HTML / JS / Canvas · StatsBomb data format

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | pipeline modules — `motion/`, `calibration/`, `detection/`, `tracking/`, `analytics/`, `visualization/`, `export/` |
| `tools/` | CLI tools (auto-labeling, pipeline, assembly, nightly runner) + the review web app in `tools/review/` |
| `config/` | per-clip configs, calibrations, tracker settings |
| `ARCHITECTURE.md` | detailed, up-to-date module reference (German) |
| `PLAN.md` | prioritized roadmap and changelog |

## Running it

```bash
pip install -r requirements.txt

# rebuild the production tracking for the test clips (needs the data + model, see below)
python -m tools.xmodel_pipeline
# merge the cameras into global IDs
python -m tools.assemble_x
# interactive review tool  →  http://127.0.0.1:8731/
python -m tools.review.server
```

## Data & models not included

The raw footage shows **real people** and is intentionally **not** part of this repository
for privacy reasons; the model weights and generated outputs are excluded for size. The code
is the deliverable — point the pipeline at your own Veo-style footage and a football-trained
YOLO model.

## Status & limitations

A working prototype on ~20-second test clips, CPU-only. The main open problems — same-team
ID switches at crossings, and running a full match at scale — need a re-identification model
and GPU acceleration. The current state and next steps are tracked in [`PLAN.md`](PLAN.md).
