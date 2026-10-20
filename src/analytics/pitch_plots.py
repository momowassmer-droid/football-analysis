"""
Pitch visualizations: team heatmaps and player run paths.

Consumes a detections CSV with pitch_x / pitch_y (StatsBomb coordinates)
and renders analysis graphics on a proper football pitch via mplsoccer.
These are the first "fan-facing" outputs the pipeline produces.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no GUI window
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch


def team_heatmap(
    detections_csv: Path,
    output_png: Path,
    team_id: int,
) -> None:
    """Density heatmap of one team's positions over the clip."""

    df = pd.read_csv(detections_csv)
    df = df[(df["class_name"] == "person") & (df["team_id"] == team_id)]
    df = df.dropna(subset=["pitch_x", "pitch_y"])

    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="#22312b")
    fig, ax = pitch.draw(figsize=(10, 6.5))
    pitch.kdeplot(df["pitch_x"], df["pitch_y"], ax=ax, fill=True,
                  levels=50, cmap="hot", alpha=0.7)
    ax.set_title(f"Team {team_id} - Heatmap", color="white")
    fig.set_facecolor("#22312b")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches="tight", facecolor="#22312b")
    plt.close(fig)
    print(f"Wrote heatmap -> {output_png}")


def run_paths(
    detections_csv: Path,
    output_png: Path,
    min_track_frames: int = 20,
    max_gap: int = 10,
) -> None:
    """Trajectory lines for each sufficiently long player track.

    Lines are broken wherever a track has a frame gap larger than ``max_gap``:
    a player undetected for a while is not actually running in a straight line
    between where they vanished and reappeared, so connecting those points
    would draw a phantom straight segment across the pitch.
    """

    import numpy as np

    df = pd.read_csv(detections_csv)
    df = df[(df["class_name"] == "person")].dropna(subset=["track_id", "pitch_x", "pitch_y"])

    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="#22312b")
    fig, ax = pitch.draw(figsize=(10, 6.5))

    cmap = plt.get_cmap("tab20")
    for i, (tid, g) in enumerate(df.groupby("track_id")):
        if len(g) < min_track_frames:
            continue
        g = g.sort_values("frame")
        # Insert a NaN wherever the frame gap is too large, so plot lifts the
        # pen there instead of drawing a straight connector across the gap.
        x = g["pitch_x"].to_numpy(dtype=float).copy()
        y = g["pitch_y"].to_numpy(dtype=float).copy()
        gap_after = np.diff(g["frame"].to_numpy()) > max_gap
        x[:-1][gap_after] = np.nan
        y[:-1][gap_after] = np.nan
        pitch.plot(x, y, ax=ax, lw=2, color=cmap(i % 20), alpha=0.8)

    ax.set_title("Player run paths", color="white")
    fig.set_facecolor("#22312b")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches="tight", facecolor="#22312b")
    plt.close(fig)
    print(f"Wrote run paths -> {output_png}")


TEAM_PLOT_COLORS = {0: "#2ec4ff", 1: "#ff4d4d", 2: "#2ee65a", 3: "#ff00ff"}


def formation_map(avg_positions_csv: Path, output_png: Path) -> None:
    """Average position of each player on the pitch, colored by team (formation)."""

    df = pd.read_csv(avg_positions_csv)

    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="#22312b")
    fig, ax = pitch.draw(figsize=(10, 6.5))

    for r in df.itertuples(index=False):
        color = TEAM_PLOT_COLORS.get(int(r.team_id), "#cccccc")
        pitch.scatter(r.pitch_x, r.pitch_y, ax=ax, s=420, color=color,
                      edgecolors="black", linewidth=1.2, zorder=3)
        ax.text(r.pitch_x, r.pitch_y, str(int(r.track_id)), color="black",
                ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)

    ax.set_title("Average positions (formation)", color="white")
    fig.set_facecolor("#22312b")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches="tight", facecolor="#22312b")
    plt.close(fig)
    print(f"Wrote formation map -> {output_png}")
