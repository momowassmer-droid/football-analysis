"""
Event detection from a possession sequence.

Turns the per-frame possession track into discrete football events by
finding where the ball changes hands:

  * same team, new player      -> pass
  * different team, new player  -> turnover (interception / recovery)

Short possession flickers are ignored (a spell must last a minimum number
of frames) so detector noise doesn't create spurious passes.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

MIN_SPELL_FRAMES = 3


@dataclass(slots=True)
class Event:
    frame: int
    type: str               # "pass" | "turnover"
    team_id: int
    from_track: int
    to_track: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float


def _possession_spells(per_frame: pd.DataFrame, min_frames: int) -> list[dict]:
    """Collapse consecutive same-owner frames into spells, dropping short ones."""
    spells = []
    current = None
    for _, row in per_frame.sort_values("frame").iterrows():
        tid = int(row["track_id"])
        if current is None or tid != current["track_id"]:
            if current is not None and current["frames"] >= min_frames:
                spells.append(current)
            current = {
                "track_id": tid,
                "team_id": int(row["team_id"]),
                "start_frame": int(row["frame"]),
                "x": float(row["pitch_x"]),
                "y": float(row["pitch_y"]),
                "frames": 1,
            }
        else:
            current["frames"] += 1
            current["x"], current["y"] = float(row["pitch_x"]), float(row["pitch_y"])
    if current is not None and current["frames"] >= min_frames:
        spells.append(current)
    return spells


def detect_events(
    possession_csv: Path,
    output_csv: Path,
    min_spell_frames: int = MIN_SPELL_FRAMES,
) -> list[Event]:

    event_columns = ["frame", "type", "team_id", "from_track", "to_track",
                     "start_x", "start_y", "end_x", "end_y"]

    per_frame = pd.read_csv(possession_csv)
    if per_frame.empty:
        print("Events: no possession data")
        pd.DataFrame(columns=event_columns).to_csv(output_csv, index=False)
        return []

    spells = _possession_spells(per_frame, min_spell_frames)

    events: list[Event] = []
    for prev, nxt in zip(spells, spells[1:]):
        event_type = "pass" if prev["team_id"] == nxt["team_id"] else "turnover"
        events.append(Event(
            frame=nxt["start_frame"],
            type=event_type,
            team_id=prev["team_id"],
            from_track=prev["track_id"],
            to_track=nxt["track_id"],
            start_x=prev["x"], start_y=prev["y"],
            end_x=nxt["x"], end_y=nxt["y"],
        ))

    df = pd.DataFrame([asdict(e) for e in events], columns=event_columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    n_pass = sum(1 for e in events if e.type == "pass")
    n_turn = sum(1 for e in events if e.type == "turnover")
    print(f"Events: {len(events)} total ({n_pass} passes, {n_turn} turnovers)")

    return events
