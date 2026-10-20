"""
Load pipeline configuration from a YAML file into typed dataclasses.
"""

from dataclasses import dataclass
from dataclasses import field as dcfield
from pathlib import Path

import yaml


@dataclass(slots=True)
class DetectionConfig:
    model: str = "yolov8n.pt"
    imgsz: int = 640
    nms_iou: float = 0.7
    confidence: float = 0.25
    stabilize: bool = False   # warp each frame onto a reference before tracking
                              # (cancels camera pan/zoom -> steadier association)


@dataclass(slots=True)
class TrackingConfig:
    tracker: str = "bytetrack.yaml"


@dataclass(slots=True)
class StitchingConfig:
    enabled: bool = True
    max_frame_gap: int = 30
    max_distance_px: float = 80.0


@dataclass(slots=True)
class FieldConfig:
    enabled: bool = False
    calibration: str | None = None  # path to a PitchCalibration homography JSON (preferred)
    region: str | None = None       # path to a FieldRegion polygon JSON (fallback)
    margin: float = 2.0             # pitch-unit tolerance at the touchlines


@dataclass(slots=True)
class TrackFilterConfig:
    enabled: bool = False
    pitch_margin: float = 0.5
    min_inside_ratio: float = 0.60


@dataclass(slots=True)
class VenueConfig:
    """Where the match is played -- the 'Spielstätte'."""
    name: str = ""
    pitch_length_m: float = 105.0   # goal-to-goal (maps to StatsBomb x, 0..120)
    pitch_width_m: float = 68.0     # touchline-to-touchline (maps to y, 0..80)
    surface: str = "unbekannt"      # e.g. Rasen, Kunstrasen, Hybridrasen


@dataclass(slots=True)
class PipelineConfig:
    video_input: Path
    detections_output: Path
    detection: DetectionConfig = dcfield(default_factory=DetectionConfig)
    tracking: TrackingConfig = dcfield(default_factory=TrackingConfig)
    field: FieldConfig = dcfield(default_factory=FieldConfig)
    track_filter: TrackFilterConfig = dcfield(default_factory=TrackFilterConfig)
    stitching: StitchingConfig = dcfield(default_factory=StitchingConfig)
    venue: VenueConfig = dcfield(default_factory=VenueConfig)
    evaluate: bool = True


def load_config(config_path: Path) -> PipelineConfig:
    """
    Read ``config_path`` (YAML) and return a fully populated
    PipelineConfig. Missing keys fall back to the dataclass defaults.
    """

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    video = raw.get("video", {})
    output = raw.get("output", {})
    detection = raw.get("detection", {})
    tracking = raw.get("tracking", {})
    field_cfg = raw.get("field", {})
    track_filter = raw.get("track_filter", {})
    stitching = raw.get("stitching", {})
    venue = raw.get("venue", {})

    return PipelineConfig(
        video_input=Path(video["input"]),
        detections_output=Path(output["detections"]),
        detection=DetectionConfig(
            model=detection.get("model", DetectionConfig.model),
            imgsz=detection.get("imgsz", DetectionConfig.imgsz),
            nms_iou=detection.get("nms_iou", DetectionConfig.nms_iou),
            confidence=detection.get("confidence", DetectionConfig.confidence),
            stabilize=detection.get("stabilize", DetectionConfig.stabilize),
        ),
        tracking=TrackingConfig(
            tracker=tracking.get("tracker", TrackingConfig.tracker),
        ),
        field=FieldConfig(
            enabled=field_cfg.get("enabled", FieldConfig.enabled),
            calibration=field_cfg.get("calibration", FieldConfig.calibration),
            region=field_cfg.get("region", FieldConfig.region),
            margin=field_cfg.get("margin", FieldConfig.margin),
        ),
        track_filter=TrackFilterConfig(
            enabled=track_filter.get("enabled", TrackFilterConfig.enabled),
            pitch_margin=track_filter.get("pitch_margin", TrackFilterConfig.pitch_margin),
            min_inside_ratio=track_filter.get(
                "min_inside_ratio", TrackFilterConfig.min_inside_ratio
            ),
        ),
        stitching=StitchingConfig(
            enabled=stitching.get("enabled", StitchingConfig.enabled),
            max_frame_gap=stitching.get("max_frame_gap", StitchingConfig.max_frame_gap),
            max_distance_px=stitching.get("max_distance_px", StitchingConfig.max_distance_px),
        ),
        venue=VenueConfig(
            name=venue.get("name", VenueConfig.name),
            pitch_length_m=venue.get("pitch_length_m", VenueConfig.pitch_length_m),
            pitch_width_m=venue.get("pitch_width_m", VenueConfig.pitch_width_m),
            surface=venue.get("surface", VenueConfig.surface),
        ),
        evaluate=raw.get("evaluate", True),
    )
