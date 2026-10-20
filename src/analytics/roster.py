"""
Player roster: suggest a team + name per track, let the user adjust.

Workflow:
  1. generate_roster() writes a roster JSON with a *suggestion* for every
     track (team from clustering, a placeholder name) plus a cropped image
     of each player so the user can tell who is who.
  2. The user edits the JSON -- correcting teams and typing real names.
  3. apply_roster() writes player_name (and corrected team_id) back onto the
     detections, so the renderer and analytics use real identities.
"""

import json
from pathlib import Path

import cv2
import pandas as pd

from src.video.video_reader import VideoReader

DEFAULT_TEAM_NAMES = {0: "Team A", 1: "Team B", 3: "Schiedsrichter"}


def _representative_rows(persons: pd.DataFrame) -> dict[int, pd.Series]:
    """For each track, the detection with the largest (clearest) box."""
    reps: dict[int, pd.Series] = {}
    persons = persons.assign(_area=(persons["x2"] - persons["x1"]) * (persons["y2"] - persons["y1"]))
    for tid, g in persons.groupby("track_id"):
        reps[int(tid)] = g.loc[g["_area"].idxmax()]
    return reps


def generate_roster(
    detections_csv: Path,
    video_path: Path,
    output_json: Path,
    crops_dir: Path,
    team_names: dict[int, str] | None = None,
) -> None:
    """Write a roster JSON of team+name suggestions, with a crop per player."""

    team_names = team_names or DEFAULT_TEAM_NAMES

    df = pd.read_csv(detections_csv)
    persons = df[(df["class_name"] == "person") & df["track_id"].notna()].copy()
    persons["track_id"] = persons["track_id"].astype(int)

    reps = _representative_rows(persons)
    team_of = {tid: int(g["team_id"].dropna().mode().iloc[0]) if g["team_id"].notna().any() else -1
               for tid, g in persons.groupby("track_id")}
    frame_count = {tid: int(len(g)) for tid, g in persons.groupby("track_id")}

    # extract one crop per track at its representative frame
    crops_dir.mkdir(parents=True, exist_ok=True)
    needed = {int(r["frame"]): tid for tid, r in reps.items()}
    crop_paths: dict[int, str] = {}

    reader = VideoReader(video_path)
    reader.open()
    for frame in reader.frames():
        tid = needed.get(frame.frame_id)
        if tid is None:
            continue
        r = reps[tid]
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        crop = frame.image[max(y1, 0):y2, max(x1, 0):x2]
        path = crops_dir / f"track_{tid}.jpg"
        if crop.size:
            cv2.imwrite(str(path), crop)
            crop_paths[tid] = str(path)
    reader.close()

    players = [
        {
            "track_id": tid,
            "team": team_of.get(tid, -1),
            "name": f"Spieler {tid}",            # suggestion; user replaces
            "frames": frame_count.get(tid, 0),
            "crop": crop_paths.get(tid, ""),
        }
        for tid in sorted(reps.keys())
    ]

    roster = {"team_names": {str(k): v for k, v in team_names.items()}, "players": players}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)

    print(f"Roster suggestion: {len(players)} players -> {output_json}")
    print(f"Crops for identification -> {crops_dir}")
    print("Edit the JSON (name + team), then run apply_roster().")


def apply_roster(
    detections_csv: Path,
    roster_json: Path,
    output_csv: Path,
) -> None:
    """Apply the (edited) roster: write player_name and corrected team_id."""

    with roster_json.open("r", encoding="utf-8") as f:
        roster = json.load(f)

    name_map = {int(p["track_id"]): p["name"] for p in roster["players"]}
    team_map = {int(p["track_id"]): int(p["team"]) for p in roster["players"]}

    df = pd.read_csv(detections_csv)

    def _name(tid):
        return name_map.get(int(tid)) if pd.notna(tid) else None

    def _team(tid, current):
        if pd.isna(tid):
            return current
        return team_map.get(int(tid), current)

    df["player_name"] = df["track_id"].map(_name)
    df["team_id"] = df.apply(lambda r: _team(r["track_id"], r.get("team_id")), axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Applied roster: {len(name_map)} named players -> {output_csv}")
