"""
End-to-end analysis workflow: video -> data -> analyses -> visualizations.

Chains every stage built in the project into one reproducible run, driven
by the config file, writing a tidy per-clip output tree:

    data/output/<clip>/
        detections/   raw + processed CSVs (pitch coords, teams, ball)
        analytics/    possession, events, StatsBomb JSON, heatmaps, run paths
        video/        annotated tracking video
        roster/       team+name suggestions + player crops

Stages skipped automatically when their prerequisites (e.g. a calibration)
are missing, so a clip works before it is fully set up.
"""

from dataclasses import dataclass
from pathlib import Path

from src.analytics.events import detect_events
from src.analytics.physical_metrics import average_positions, player_distances, team_compactness
from src.analytics.pitch_plots import formation_map, run_paths, team_heatmap
from src.analytics.possession import compute_possession
from src.analytics.roster import generate_roster
from src.analytics.team_classifier import classify_teams
from src.calibration.homography import PitchCalibration, filter_to_pitch_csv
from src.config.config_loader import PipelineConfig, load_config
from src.detection.ball_search_tracker import track_ball_search
from src.detection.ball_lowconf import detect_ball_lowconf
from src.export.statsbomb_exporter import export_events_statsbomb
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.tracking.offfield_filter import filter_offfield_tracks_csv
from src.tracking.smoothing import smooth_tracks_csv
from src.tracking.enhanced_stitcher import stitch_csv_enhanced
from src.visualization.team_ball_renderer import render_teams_and_ball


@dataclass(slots=True)
class WorkflowPaths:
    root: Path
    detections: Path
    analytics: Path
    video: Path
    roster: Path

    @classmethod
    def for_clip(cls, output_root: Path, clip_name: str) -> "WorkflowPaths":
        root = output_root / clip_name
        p = cls(root, root / "detections", root / "analytics", root / "video", root / "roster")
        for d in (p.detections, p.analytics, p.video, p.roster):
            d.mkdir(parents=True, exist_ok=True)
        return p


def _write_venue(venue, path: Path) -> None:
    """Write the Spielstätte (venue) info alongside the analytics."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": venue.name,
        "pitch_length_m": venue.pitch_length_m,
        "pitch_width_m": venue.pitch_width_m,
        "surface": venue.surface,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Spielstätte: {venue.name or '(unbenannt)'} | "
          f"{venue.pitch_length_m}x{venue.pitch_width_m} m | {venue.surface}")


def run_workflow(config: PipelineConfig, calibration_path: Path | None = None,
                 n_teams: int = 2, ball_model: str = "yolov8n.pt", force: bool = False) -> WorkflowPaths:
    video = config.video_input
    clip = video.stem
    paths = WorkflowPaths.for_clip(Path("data/output"), clip)
    D, A, V = paths.detections, paths.analytics, paths.video

    print(f"\n===== WORKFLOW: {clip} =====")

    # 1. detection + tracking (config-driven: model, imgsz, tracker) -- cached,
    # since this is the expensive CPU step; pass force=True to redo it.
    raw = D / "detections_raw.csv"
    if force or not raw.exists():
        # postprocess=False: this workflow does its own field-filter/stitch
        # below with better methods (motion-compensated calibration, the
        # enhanced stitcher), so the pipeline's built-in legacy postprocess
        # would otherwise filter/stitch the same data twice over.
        AnalysisPipeline.from_config(config).run(video, raw, postprocess=False)
    else:
        print(f"  detection cached -> {raw}")

    has_calib = calibration_path is not None and Path(calibration_path).exists()
    if not has_calib:
        print("\nNo calibration -> stopping after detection (calibrate to unlock the rest).")
        return paths

    calibration = PitchCalibration.load(calibration_path)

    # Shared camera-motion track (per-frame projective homography onto the
    # reference frame): the pitch filter below computes and caches it once, so
    # the renderer (step 7) reuses it instead of re-estimating camera motion.
    motion_cache = D / "camera_motion.npz"

    # 2. calibration: pitch coords + on-pitch filter (also cached), then clean tracks
    pitch = D / "pitch.csv"
    if force or not pitch.exists():
        filter_to_pitch_csv(raw, pitch, calibration, margin=config.field.margin,
                            video_path=video, motion_cache=motion_cache)
    clean = D / "clean.csv"
    if force or not clean.exists():
        filter_offfield_tracks_csv(pitch, clean)
    # Real fps drives the speed gates in stitching/smoothing (a wrong default
    # mis-scales "physically possible" motion). Read it once from the clip.
    fps = _video_fps(video)
    stitched = D / "stitched.csv"
    stitch_csv_enhanced(video, clean, stitched, fps=fps)
    smooth = D / "tracks.csv"
    smooth_tracks_csv(stitched, smooth, fps=fps)

    # 3. teams + ball
    teams = D / "teams.csv"
    classify_teams(video, smooth, teams, n_clusters=n_teams)
    ball_cands = D / "ball_candidates.csv"
    ball = D / "ball.csv"
    # The ball is detected best by yolov8n, independent of the player model.
    # Cached like detection: ball positions are pure image-space results, so
    # they stay valid when calibration/motion change and only ``force`` (or a
    # missing file) should trigger the ~10-minute redetection.
    if force or not ball.exists():
        detect_ball_lowconf(video, ball_cands, model_name=ball_model, imgsz=config.detection.imgsz)
        track_ball_search(video, ball, model_name=ball_model, init_candidates_csv=ball_cands)
    else:
        print(f"  ball cached -> {ball}")

    # 4. roster suggestion (user edits, then re-apply)
    generate_roster(teams, video, paths.roster / "roster.json", paths.roster / "crops")

    # 5. analytics: possession -> events -> StatsBomb export
    possession = A / "possession.csv"
    compute_possession(teams, ball, calibration_path, possession)
    events = A / "events.csv"
    detect_events(possession, events)
    export_events_statsbomb(events, A / "events_statsbomb.json")

    # 6. physical / tactical metrics (ball-independent), in real metres for THIS venue
    _write_venue(config.venue, A / "venue.json")
    vl, vw = config.venue.pitch_length_m, config.venue.pitch_width_m
    player_distances(teams, A / "player_distances.csv", pitch_length_m=vl, pitch_width_m=vw)
    team_compactness(teams, A / "team_compactness.csv", pitch_length_m=vl, pitch_width_m=vw)
    average_positions(teams, A / "average_positions.csv")

    # 7. visualizations
    for t in range(n_teams):
        team_heatmap(teams, A / f"team{t}_heatmap.png", team_id=t)
    run_paths(teams, A / "run_paths.png")
    formation_map(A / "average_positions.csv", A / "formation.png")
    render_teams_and_ball(video, teams, ball, V / f"{clip}_tracked.mp4",
                          calibration_path=calibration_path, motion_cache=motion_cache,
                          pitch_length_m=vl, pitch_width_m=vw)

    print(f"\n===== DONE: outputs under {paths.root} =====")
    return paths


def _video_fps(video: Path, default: float = 25.0) -> float:
    """Frame rate of a clip, for the physical speed gates. Falls back to default."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1 else default


def _load_manual_links(path: Path | None) -> list[list] | None:
    import json
    if path and path.exists():
        links = json.loads(path.read_text(encoding="utf-8"))
        print(f"Manual links loaded: {len(links)} user-specified merges")
        return links
    return None


def _assemble(teams_csvs: dict[str, Path], match_name: str, manual_links_json: Path | None,
              venue=None) -> Path:
    """Assemble already-processed half views (fast: no detection)."""
    from src.analytics.match_assembler import assemble_match
    from src.analytics.pitch_plots import run_paths
    from src.config.config_loader import VenueConfig

    venue = venue or VenueConfig()

    teams_csvs = {v: p for v, p in teams_csvs.items() if p.exists()}
    if len(teams_csvs) < 2:
        print(f"Only {len(teams_csvs)} calibrated view(s) -> nothing to assemble.")
        return Path("data/output") / match_name

    out = Path("data/output") / match_name
    (out / "analytics").mkdir(parents=True, exist_ok=True)
    A = out / "analytics"
    merged = A / "match_tracks.csv"
    assemble_match(teams_csvs, merged, manual_links=_load_manual_links(manual_links_json))
    run_paths(merged, A / "match_run_paths.png")

    # match-level physical / tactical metrics, in real metres for THIS venue
    _write_venue(venue, A / "venue.json")
    vl, vw = venue.pitch_length_m, venue.pitch_width_m
    player_distances(merged, A / "match_player_distances.csv", pitch_length_m=vl, pitch_width_m=vw)
    team_compactness(merged, A / "match_team_compactness.csv", pitch_length_m=vl, pitch_width_m=vw)
    average_positions(merged, A / "match_average_positions.csv")
    formation_map(A / "match_average_positions.csv", A / "match_formation.png")

    print(f"\n===== MATCH ASSEMBLED: {out} =====")
    return out


def run_match(half_configs: list[Path], match_name: str = "match",
              manual_links_json: Path | None = None) -> Path:
    """
    Full multi-half match: run each half's workflow, then assemble all halves
    into one global-id dataset. Views are matched across all pairs (order-free).
    ``manual_links_json`` (optional) lets the user force-merge tracks the auto
    matcher missed -- a list of groups like [[["A",5],["B",12]], ...].
    """
    teams_csvs = {}
    venue = None
    for i, cfg_path in enumerate(half_configs):
        config = load_config(cfg_path)
        venue = venue or config.venue
        calib = Path(config.field.calibration) if config.field.calibration else None
        paths = run_workflow(config, calibration_path=calib)
        teams_csvs[chr(ord("A") + i)] = paths.detections / "teams.csv"
    return _assemble(teams_csvs, match_name, manual_links_json, venue=venue)


def _load_venue(clip: str):
    """Recover a clip's Spielstätte from the venue.json its own run_workflow wrote."""
    import json

    from src.config.config_loader import VenueConfig

    path = Path("data/output") / clip / "analytics" / "venue.json"
    if not path.exists():
        return VenueConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return VenueConfig(
        name=data.get("name", VenueConfig.name),
        pitch_length_m=data.get("pitch_length_m", VenueConfig.pitch_length_m),
        pitch_width_m=data.get("pitch_width_m", VenueConfig.pitch_width_m),
        surface=data.get("surface", VenueConfig.surface),
    )


def reassemble_match(half_clips: dict[str, str], match_name: str = "match",
                     manual_links_json: Path | None = None) -> Path:
    """
    Re-assemble already-processed halves (e.g. after editing manual links) WITHOUT
    re-running detection. ``half_clips`` maps a view label to a clip name, e.g.
    {"A": "match1_left", "B": "match1_right"}.
    """
    teams_csvs = {v: Path("data/output") / clip / "detections" / "teams.csv"
                  for v, clip in half_clips.items()}
    venue = _load_venue(next(iter(half_clips.values())))
    return _assemble(teams_csvs, match_name, manual_links_json, venue=venue)


def main() -> None:
    import sys
    args = sys.argv[1:]
    if len(args) >= 2 and all(a.endswith((".yaml", ".yml")) for a in args):
        run_match([Path(a) for a in args])          # multiple configs -> match assembly
        return
    config = load_config(Path(args[0]) if args else Path("src/config/config.yaml"))
    calib = Path(config.field.calibration) if config.field.calibration else None
    run_workflow(config, calibration_path=calib)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows: prevent spawn re-import recursion
    main()
