"""
Remove off-field people (spectators, coaches) that survive the pitch filter.

Spectators in this venue stand right behind the far touchline, so they map
to pitch_y ~= 0 -- geometrically inside the pitch. Two grounded rules,
asymmetric because the two failure modes differ:

  A) far-line band: a track whose mean position sits in the far strip
     (pitch_y < far_band) is a spectator. Position is used, not movement,
     because perspective noise near the far line inflates apparent motion.
     Cost: the far ~far_band units of the pitch are sacrificed.

  B) static edge person: a track near any other boundary that barely moves
     is staff (e.g. a coach on the near sideline). Real players there move,
     so movement separates them; position alone would delete real players.

Needs pitch_x / pitch_y columns. Operates per track; non-person rows pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def filter_offfield_tracks_csv(
    input_csv: Path,
    output_csv: Path,
    far_band: float = 10.0,
    edge_band: float = 4.0,
    static_move: float = 1.5,
    min_frames: int = 3,
    class_name: str = "person",
    far_static_only: bool = False,
) -> None:
    """See module docstring. ``far_static_only=True`` makes rule A also require
    the far-line track to barely move, so a real player playing near the far
    touchline (who moves several pitch units) is kept -- important for the
    x-model pipeline where far-half coverage matters. Measured on this footage:
    real far-line players move ~4-6 units, static spectators ~0, so the movement
    guard separates them cleanly. Default False preserves the legacy behavior.
    """

    df = pd.read_csv(input_csv)
    if "pitch_x" not in df.columns:
        raise ValueError("Off-field filter needs pitch coordinates (run calibration filter first).")

    persons = df[(df["class_name"] == class_name) & df["track_id"].notna()]

    keep_ids: set[int] = set()
    dropped: dict[str, int] = {"far_spectator": 0, "static_edge": 0, "too_short": 0}

    for tid, g in persons.groupby("track_id"):
        mean_x, mean_y = g["pitch_x"].mean(), g["pitch_y"].mean()
        move = float(np.hypot(g["pitch_x"].max() - g["pitch_x"].min(),
                              g["pitch_y"].max() - g["pitch_y"].min()))

        if len(g) < min_frames:
            dropped["too_short"] += 1
            continue

        # A) far touchline band -> spectators. With far_static_only, only static
        # ones are dropped, so a real player near the far line (who moves) stays.
        if mean_y < far_band and (not far_static_only or move < static_move):
            dropped["far_spectator"] += 1
            continue

        # B) static person hugging another boundary -> staff/coach
        near_edge = (
            mean_y > PITCH_WIDTH - edge_band
            or mean_x < edge_band
            or mean_x > PITCH_LENGTH - edge_band
        )
        if near_edge and move < static_move:
            dropped["static_edge"] += 1
            continue

        keep_ids.add(int(tid))

    is_person = df["class_name"] == class_name
    keep_mask = (~is_person) | df["track_id"].isin(keep_ids)
    kept = df[keep_mask].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_csv, index=False)

    print(
        f"Off-field filter: kept {len(keep_ids)} tracks, dropped "
        f"{dropped['far_spectator']} far-line spectators, "
        f"{dropped['static_edge']} static edge staff, "
        f"{dropped['too_short']} too-short"
    )
