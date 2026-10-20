"""
Compare the current vs the enhanced tracking stack on one detections CSV.

Both stacks start from the same raw tracked detections so the difference
is purely the tracking post-processing, not the detector.

Stack A (current):  calib-filter -> track-filter -> pixel stitch
Stack B (enhanced): calib-filter -> track-filter -> metric/color stitch
                    -> gap interpolation -> trajectory smoothing

Usage:
    python -m tools.compare_tracking <video> <raw_detections_csv> <calibration_json> <out_prefix>
"""

import sys
from pathlib import Path

from src.analytics.team_classifier import classify_teams
from src.calibration.homography import PitchCalibration, filter_to_pitch_csv
from src.detection.ball_extractor import extract_ball_track
from src.tracking.enhanced_stitcher import stitch_csv_enhanced
from src.tracking.interpolation import interpolate_tracks_csv
from src.tracking.smoothing import smooth_tracks_csv
from src.tracking.track_filter import filter_tracks_csv
from src.tracking.track_stitcher import stitch_csv
from src.visualization.team_ball_renderer import render_teams_and_ball

DET = Path("data/output/detections")
VID = Path("data/output/videos")


def _common_prefilter(raw_csv: Path, calib: PitchCalibration, tag: str) -> Path:
    """calibration pitch-filter (+ coords) -> track filter. Shared by both stacks."""
    pitch = DET / f"{tag}_pitch.csv"
    filt = DET / f"{tag}_trackfilt.csv"
    filter_to_pitch_csv(raw_csv, pitch, calib, margin=2.0)
    filter_tracks_csv(pitch, filt)
    return filt


def _finish(video: Path, tracks_csv: Path, ball_src: Path, calib_path: Path, tag: str) -> Path:
    """team classification + ball + render with field overlay."""
    teams = DET / f"{tag}_teams.csv"
    ball = DET / f"{tag}_ball.csv"
    out = VID / f"{tag}.mp4"
    classify_teams(video, tracks_csv, teams)
    extract_ball_track(ball_src, ball)
    render_teams_and_ball(video, teams, ball, out, calibration_path=calib_path)
    return out


def stack_current(video: Path, raw_csv: Path, calib_path: Path, prefix: str) -> Path:
    calib = PitchCalibration.load(calib_path)
    filt = _common_prefilter(raw_csv, calib, f"{prefix}_A")
    stitched = DET / f"{prefix}_A_stitched.csv"
    stitch_csv(filt, stitched)
    return _finish(video, stitched, filt, calib_path, f"{prefix}_A_current")


def stack_improved(video: Path, raw_csv: Path, calib_path: Path, prefix: str) -> Path:
    calib = PitchCalibration.load(calib_path)
    filt = _common_prefilter(raw_csv, calib, f"{prefix}_B")
    stitched = DET / f"{prefix}_B_stitched.csv"
    interp = DET / f"{prefix}_B_interp.csv"
    smooth = DET / f"{prefix}_B_smooth.csv"
    stitch_csv_enhanced(video, filt, stitched)
    interpolate_tracks_csv(stitched, interp)
    smooth_tracks_csv(interp, smooth)
    return _finish(video, smooth, filt, calib_path, f"{prefix}_B_improved")


def main() -> None:
    if len(sys.argv) < 5:
        print(__doc__)
        return
    video, raw_csv, calib_path, prefix = (
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    )
    print("\n########## STACK A (current) ##########")
    a = stack_current(video, raw_csv, calib_path, prefix)
    print("\n########## STACK B (improved) ##########")
    b = stack_improved(video, raw_csv, calib_path, prefix)
    print(f"\nVideo A (current):  {a}")
    print(f"Video B (improved): {b}")


if __name__ == "__main__":
    main()
