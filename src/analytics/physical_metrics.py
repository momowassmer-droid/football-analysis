"""
Ball-independent physical & tactical metrics from tracking data.

Everything here needs only player positions in pitch coordinates -- no ball,
no events -- so it is reliable even while ball tracking is weak. Works on a
per-clip tracks.csv or on the merged match_tracks.csv (same columns:
track_id, team_id, pitch_x, pitch_y, class_name).

Pitch coordinates (StatsBomb 120 x 80) are converted to metres using a
standard 105 x 68 m pitch so distances and speeds are real-world.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0
MAX_SPEED_MS = 12.0        # cap: faster steps are tracking errors, not running
SPRINT_THRESHOLD_MS = 5.5  # ~20 km/h
MAX_GAP_FRAMES = 10        # don't measure across longer holes


def _to_metres(df: pd.DataFrame, pitch_length_m: float, pitch_width_m: float) -> pd.DataFrame:
    """Convert StatsBomb pitch units (120 x 80) to metres for THIS venue's pitch."""
    df = df[(df["class_name"] == "person") & df["track_id"].notna()].copy()
    df["mx"] = df["pitch_x"] * (pitch_length_m / 120.0)
    df["my"] = df["pitch_y"] * (pitch_width_m / 80.0)
    return df


def player_distances(tracks_csv: Path, output_csv: Path, fps: float = 30.0,
                     speed_window: int = 7,
                     pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
                     pitch_width_m: float = DEFAULT_PITCH_WIDTH_M) -> pd.DataFrame:
    """
    Per player: distance covered (frame-to-frame, outlier-capped) plus speeds
    measured over a ``speed_window``-frame window (~0.25 s) so single-frame
    jitter cannot fake impossible top speeds.
    """

    df = _to_metres(pd.read_csv(tracks_csv), pitch_length_m, pitch_width_m)
    rows = []

    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy()
        mx, my = g["mx"].to_numpy(), g["my"].to_numpy()

        # distance: sum of plausible frame-to-frame steps
        step = np.hypot(np.diff(mx), np.diff(my))
        dfr = np.diff(fr)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (dfr >= 1) & (dfr <= MAX_GAP_FRAMES) & (step / (dfr / fps) <= MAX_SPEED_MS)
        distance = float(np.nansum(step[ok]))

        # speed: displacement over a multi-frame window (robust to jitter)
        speeds = np.array([])
        if len(fr) > speed_window:
            w = speed_window
            dxw, dyw = mx[w:] - mx[:-w], my[w:] - my[:-w]
            dfrw = fr[w:] - fr[:-w]
            with np.errstate(invalid="ignore", divide="ignore"):
                sp = np.hypot(dxw, dyw) / (dfrw / fps)
            speeds = sp[(dfrw <= w + MAX_GAP_FRAMES) & (sp <= MAX_SPEED_MS)]

        team = g["team_id"].dropna()
        rows.append({
            "track_id": int(tid),
            "team_id": int(team.mode().iloc[0]) if len(team) else -1,
            "frames": int(len(g)),
            "distance_m": round(distance, 1),
            "top_speed_kmh": round(float(np.nanmax(speeds)) * 3.6, 1) if len(speeds) else 0.0,
            "avg_speed_kmh": round(float(np.nanmean(speeds)) * 3.6, 1) if len(speeds) else 0.0,
            "high_speed_pct": round(100 * float(np.mean(speeds > SPRINT_THRESHOLD_MS)), 1) if len(speeds) else 0.0,
        })

    out = pd.DataFrame(rows).sort_values("distance_m", ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Player distances: {len(out)} players "
          f"(max {out['distance_m'].max():.0f} m, top speed {out['top_speed_kmh'].max():.1f} km/h)")
    return out


def team_compactness(tracks_csv: Path, output_csv: Path,
                     pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
                     pitch_width_m: float = DEFAULT_PITCH_WIDTH_M) -> pd.DataFrame:
    """Per team: width, depth and spread (compactness) averaged over the clip."""

    df = _to_metres(pd.read_csv(tracks_csv), pitch_length_m, pitch_width_m)
    rows = []

    for (frame, team), g in df.groupby(["frame", "team_id"]):
        if len(g) < 3 or team < 0:
            continue
        cx, cy = g["mx"].mean(), g["my"].mean()
        spread = float(np.hypot(g["mx"] - cx, g["my"] - cy).mean())
        rows.append({
            "frame": int(frame), "team_id": int(team),
            "width_m": float(g["my"].max() - g["my"].min()),
            "depth_m": float(g["mx"].max() - g["mx"].min()),
            "spread_m": spread,
        })

    per_frame = pd.DataFrame(rows)
    summary = (per_frame.groupby("team_id")[["width_m", "depth_m", "spread_m"]]
               .mean().round(1).reset_index())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print("Team compactness (mean over clip):")
    for r in summary.itertuples(index=False):
        print(f"  Team {int(r.team_id)}: width {r.width_m} m, depth {r.depth_m} m, spread {r.spread_m} m")
    return summary


def average_positions(tracks_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Per player: mean pitch position (for a formation map). Keeps StatsBomb units."""

    df = pd.read_csv(tracks_csv)
    df = df[(df["class_name"] == "person") & df["track_id"].notna()]
    out = (df.groupby("track_id")
           .agg(team_id=("team_id", lambda s: int(s.dropna().mode().iloc[0]) if s.notna().any() else -1),
                pitch_x=("pitch_x", "mean"), pitch_y=("pitch_y", "mean"),
                frames=("frame", "count"))
           .reset_index())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Average positions: {len(out)} players (for formation map)")
    return out
