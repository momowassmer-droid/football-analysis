"""
Ball possession from player and ball positions in pitch space.

For each frame with a known ball position, the player whose foot point is
closest to the ball (within a small radius) is considered "in possession";
their team owns the ball that frame. Aggregated over the clip this yields
per-team possession share -- the first genuine match statistic the pipeline
produces, made possible by the homography (distances are metric, not pixel).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.homography import PitchCalibration

POSSESSION_RADIUS = 3.0  # pitch units (~yards); within this the ball is "at the player"


@dataclass(slots=True)
class PossessionResult:
    per_frame: pd.DataFrame              # frame, track_id, team_id, distance
    team_share: dict[int, float]         # team_id -> fraction of contested frames
    contested_frames: int


def _ball_to_pitch(ball: pd.DataFrame, calibration: PitchCalibration) -> pd.DataFrame:
    pts = ball[["x", "y"]].to_numpy().reshape(-1, 1, 2).astype(np.float32)
    import cv2
    mapped = cv2.perspectiveTransform(pts, calibration.homography).reshape(-1, 2)
    out = ball[["frame"]].copy()
    out["ball_x"], out["ball_y"] = mapped[:, 0], mapped[:, 1]
    return out


def compute_possession(
    players_csv: Path,
    ball_csv: Path,
    calibration_path: Path,
    output_csv: Path,
    radius: float = POSSESSION_RADIUS,
) -> PossessionResult:

    calibration = PitchCalibration.load(calibration_path)

    players = pd.read_csv(players_csv)
    players = players[(players["class_name"] == "person") & players["track_id"].notna()].copy()

    ball = pd.read_csv(ball_csv)
    ball_pitch = _ball_to_pitch(ball, calibration)
    ball_by_frame = {int(r["frame"]): (r["ball_x"], r["ball_y"]) for _, r in ball_pitch.iterrows()}

    rows = []
    for frame_id, bxy in ball_by_frame.items():
        frame_players = players[players["frame"] == frame_id]
        if frame_players.empty:
            continue
        d = np.hypot(frame_players["pitch_x"] - bxy[0], frame_players["pitch_y"] - bxy[1])
        idx = d.idxmin()
        if d[idx] > radius:
            continue  # nobody is close enough -> loose ball, no possession
        owner = frame_players.loc[idx]
        rows.append({
            "frame": frame_id,
            "track_id": int(owner["track_id"]),
            "team_id": int(owner["team_id"]) if pd.notna(owner["team_id"]) else -1,
            "pitch_x": float(owner["pitch_x"]),
            "pitch_y": float(owner["pitch_y"]),
            "distance": float(d[idx]),
        })

    # Explicit columns so an empty result still writes a header row -- a
    # zero-byte CSV would crash every downstream read_csv (EmptyDataError),
    # and "nobody was ever near the ball" is a legitimate outcome for a
    # camera covering a mostly empty third of the pitch.
    per_frame = pd.DataFrame(
        rows,
        columns=["frame", "track_id", "team_id", "pitch_x", "pitch_y", "distance"],
    )

    team_share: dict[int, float] = {}
    if not per_frame.empty:
        counts = per_frame["team_id"].value_counts(normalize=True)
        team_share = {int(k): float(v) for k, v in counts.items()}

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_frame.to_csv(output_csv, index=False)

    print(f"Possession: {len(per_frame)} contested frames")
    for team_id, share in sorted(team_share.items()):
        print(f"  Team {team_id}: {share:.1%}")

    return PossessionResult(
        per_frame=per_frame,
        team_share=team_share,
        contested_frames=len(per_frame),
    )
