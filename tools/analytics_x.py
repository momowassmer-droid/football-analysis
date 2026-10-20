"""Nr. 2 (Phase 1): run the ball-free analytics on the assembled global x-tracks.

This is the first piece of "closing the gap" -- the good x-model tracks finally
feed the existing analytics/plot back-end. Everything here needs only player
positions in pitch coordinates (no ball), so it works today:

    run paths, team heatmaps, player distances, team compactness, formation.

Ball-dependent outputs (possession, events, StatsBomb events) wait for the ball.

Input : data/output/review_assembled/x_match_tracks.csv  (global_id, team_id, pitch_x/y)
Output: data/output/review_assembled/analytics/

Usage:  python -m tools.analytics_x
"""

from __future__ import annotations

from pathlib import Path

from src.analytics.physical_metrics import (
    average_positions, player_distances, team_compactness,
)
from src.analytics.pitch_plots import formation_map, run_paths, team_heatmap

ROOT = Path(".")
SRC = ROOT / "data" / "output" / "review_assembled" / "x_match_tracks.csv"
OUT = ROOT / "data" / "output" / "review_assembled" / "analytics"

# Henriette-Herz-Ring venue (real pitch metres) -> honest distances/speeds.
PITCH_L, PITCH_W = 99.0, 64.0
FPS = 30.0


def main() -> None:
    if not SRC.exists():
        print(f"Fehlt: {SRC} -- erst `python -m tools.assemble_x` laufen lassen.")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    # visual (spatial, robust to fragmentation)
    run_paths(SRC, OUT / "match_run_paths.png")
    for team in (0, 1):
        team_heatmap(SRC, OUT / f"team{team}_heatmap.png", team_id=team)

    # tabular (per global player / per team)
    player_distances(SRC, OUT / "player_distances.csv", fps=FPS,
                     pitch_length_m=PITCH_L, pitch_width_m=PITCH_W)
    team_compactness(SRC, OUT / "team_compactness.csv",
                     pitch_length_m=PITCH_L, pitch_width_m=PITCH_W)
    average_positions(SRC, OUT / "average_positions.csv")
    formation_map(OUT / "average_positions.csv", OUT / "formation.png")

    print(f"Analytik geschrieben nach {OUT}")
    print("  Bilder: match_run_paths.png, team0/1_heatmap.png, formation.png")
    print("  Tabellen: player_distances.csv, team_compactness.csv, average_positions.csv")
    print("Hinweis: pro-Spieler-Werte sind noch durch Track-Fragmentierung begrenzt")
    print("(ein global_id deckt oft nur einen Teil des Spiels ab).")


if __name__ == "__main__":
    main()
