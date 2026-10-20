"""
Render team-colored player boxes plus the extracted ball position.

Validation tool for team_classifier and ball_extractor: colors each
player box by its assigned team_id (so a clustering mistake is visible
as a player in the "wrong" color) and draws a marker at the ball's
extracted position.
"""

from pathlib import Path

import cv2
import pandas as pd

from src.calibration.homography import PitchCalibration
from src.motion.camera_motion import camera_motion_for
from src.video.video_reader import VideoReader
from src.visualization.pitch_overlay import draw_pitch_lines

TEAM_COLORS = {
    0: (255, 200, 0),    # cyan-ish
    1: (0, 0, 255),      # red
    2: (0, 220, 0),      # green
    3: (255, 0, 255),    # magenta
}
BALL_COLOR = (0, 255, 255)  # yellow


def render_teams_and_ball(
    video_path: Path,
    detections_csv: Path,
    ball_csv: Path,
    output_path: Path,
    calibration_path: Path | None = None,
    motion_compensate: bool = True,
    motion_cache: Path | None = None,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> None:

    df = pd.read_csv(detections_csv)
    by_frame = {fid: g for fid, g in df.groupby("frame")}

    ball = pd.read_csv(ball_csv)
    ball_by_frame = {int(row["frame"]): row for _, row in ball.iterrows()}

    calibration = PitchCalibration.load(calibration_path) if calibration_path else None
    # Reuses the same per-frame camera-motion track already computed for the
    # pitch-coordinate mapping (or computes and caches it here) instead of
    # running its own estimation pass.
    track = (camera_motion_for(video_path, motion_cache)
             if motion_compensate and calibration else None)

    reader = VideoReader(video_path)
    reader.open()

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        reader.metadata.fps,
        (reader.metadata.width, reader.metadata.height),
    )

    for frame in reader.frames():

        image = frame.image

        if calibration is not None:
            image_to_pitch = (
                calibration.homography @ track.to_reference(frame.frame_id)
                if track is not None
                else calibration.homography
            )
            draw_pitch_lines(image, image_to_pitch, pitch_length_m, pitch_width_m)

        rows = by_frame.get(frame.frame_id)
        if rows is not None:
            for _, row in rows.iterrows():
                _draw_player(image, row)

        ball_row = ball_by_frame.get(frame.frame_id)
        if ball_row is not None:
            cv2.circle(image, (int(ball_row["x"]), int(ball_row["y"])), 10, BALL_COLOR, 3)
            cv2.putText(image, "BALL", (int(ball_row["x"]) + 12, int(ball_row["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLOR, 2)

        writer.write(image)

    writer.release()
    reader.close()
    print(f"Wrote {output_path}")


def _draw_player(image, row) -> None:

    if row["class_name"] != "person" or pd.isna(row.get("team_id")):
        return

    x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
    color = TEAM_COLORS.get(int(row["team_id"]), (200, 200, 200))

    name = row.get("player_name")
    label = str(name) if isinstance(name, str) and name else f"T{int(row['team_id'])}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
