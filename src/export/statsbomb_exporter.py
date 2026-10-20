"""
Export detected events to a StatsBomb-style JSON event file.

The internal coordinate system already matches StatsBomb (120 x 80), so
export is mostly a renaming into StatsBomb's event schema. This is a
pragmatic subset (type, team, player, location, timestamp, pass end
location) -- enough to be loaded by StatsBomb-aware tools like mplsoccer,
not a byte-exact reproduction of the official feed.
"""

import json
from pathlib import Path

import pandas as pd

# StatsBomb type ids for the events we currently produce.
_TYPE = {
    "pass": {"id": 30, "name": "Pass"},
    "turnover": {"id": 9, "name": "Interception"},
}


def _timestamp(frame: int, fps: float) -> str:
    seconds = frame / fps
    minute = int(seconds // 60)
    sec = seconds - minute * 60
    return f"00:{minute:02d}:{sec:06.3f}"


def export_events_statsbomb(
    events_csv: Path,
    output_json: Path,
    fps: float = 30.0,
) -> None:

    df = pd.read_csv(events_csv)
    if df.empty or "type" not in df.columns:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text("[]", encoding="utf-8")
        print(f"Exported 0 events (no events detected) -> {output_json}")
        return

    events = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        event = {
            "id": i,
            "index": i,
            "timestamp": _timestamp(int(row.frame), fps),
            "type": _TYPE.get(row.type, {"id": 0, "name": row.type}),
            "team": {"id": int(row.team_id), "name": f"Team {int(row.team_id)}"},
            "player": {"id": int(row.from_track), "name": f"Track {int(row.from_track)}"},
            "location": [round(float(row.start_x), 2), round(float(row.start_y), 2)],
        }
        if row.type == "pass":
            event["pass"] = {
                "recipient": {"id": int(row.to_track), "name": f"Track {int(row.to_track)}"},
                "end_location": [round(float(row.end_x), 2), round(float(row.end_y), 2)],
            }
        events.append(event)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Exported {len(events)} events (StatsBomb format) -> {output_json}")
