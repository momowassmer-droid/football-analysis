"""
Main analysis pipeline: video -> detection/tracking -> stitching -> export.
"""

from pathlib import Path

from tqdm import tqdm

from src.calibration.field_region import FieldRegion, filter_detections_csv
from src.calibration.homography import PitchCalibration, filter_to_pitch_csv
from src.config.config_loader import PipelineConfig
from src.detection.result_parser import parse_results
from src.detection.yolo_detector import YoloDetector
from src.evaluation.tracking_metrics import evaluate_tracking
from src.export.csv_exporter import CsvExporter
from src.tracking.track_filter import filter_tracks_csv
from src.tracking.track_stitcher import stitch_csv
from src.video.stabilizer import FrameStabilizer, unwarp_boxes
from src.video.video_reader import VideoReader


class AnalysisPipeline:

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        nms_iou: float = 0.7,
        imgsz: int = 640,
        confidence: float = 0.25,
        stabilize: bool = False,
        field_filter: bool = False,
        field_calibration: str | None = None,
        field_region: str | None = None,
        field_margin: float = 2.0,
        track_filter: bool = False,
        track_filter_pitch_margin: float = 0.5,
        track_filter_min_inside_ratio: float = 0.60,
        stitch: bool = False,
        evaluate: bool = False,
        stitch_max_frame_gap: int = 30,
        stitch_max_distance_px: float = 80.0,
    ):

        self.detector = YoloDetector(
            model_name=model_name,
            tracker=tracker,
            nms_iou=nms_iou,
            imgsz=imgsz,
            confidence=confidence,
        )

        self.stabilize = stabilize
        self.field_filter = field_filter
        self.field_calibration = field_calibration
        self.field_region = field_region
        self.field_margin = field_margin
        self.track_filter = track_filter
        self.track_filter_pitch_margin = track_filter_pitch_margin
        self.track_filter_min_inside_ratio = track_filter_min_inside_ratio
        self.stitch = stitch
        self.evaluate = evaluate
        self.stitch_max_frame_gap = stitch_max_frame_gap
        self.stitch_max_distance_px = stitch_max_distance_px

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "AnalysisPipeline":
        """Build a pipeline from a loaded PipelineConfig."""

        return cls(
            model_name=config.detection.model,
            tracker=config.tracking.tracker,
            nms_iou=config.detection.nms_iou,
            imgsz=config.detection.imgsz,
            confidence=config.detection.confidence,
            stabilize=config.detection.stabilize,
            field_filter=config.field.enabled,
            field_calibration=config.field.calibration,
            field_region=config.field.region,
            field_margin=config.field.margin,
            track_filter=config.track_filter.enabled,
            track_filter_pitch_margin=config.track_filter.pitch_margin,
            track_filter_min_inside_ratio=config.track_filter.min_inside_ratio,
            stitch=config.stitching.enabled,
            evaluate=config.evaluate,
            stitch_max_frame_gap=config.stitching.max_frame_gap,
            stitch_max_distance_px=config.stitching.max_distance_px,
        )

    def run(
        self,
        video_path: Path,
        output_csv: Path,
        postprocess: bool = True,
    ) -> None:
        """
        Pass ``postprocess=False`` when a caller (e.g. the multi-camera
        workflow) does its own field-filter/stitch/evaluate afterwards with
        better methods (motion-compensated calibration, the enhanced
        stitcher) -- running both here and there would filter/stitch the
        same data twice with two different (and partly stale) methods.
        """

        reader = VideoReader(video_path)
        reader.open()

        all_detections = []
        stabilizer: FrameStabilizer | None = None

        for frame in tqdm(
            reader.frames(),
            total=reader.metadata.frame_count,
            desc="Detecting",
            unit="frame",
        ):
            transform = None
            if self.stabilize:
                # Warp onto a fixed reference (first frame) so the tracker sees
                # only true player motion, not camera pan; boxes are mapped back
                # to original pixels below so stored data stays in image space.
                # This alignment is internal to the tracker -- the pitch mapping
                # and overlay downstream get their (more accurate, projective)
                # per-frame transform from ``camera_motion`` instead.
                if stabilizer is None:
                    stabilizer = FrameStabilizer(frame.image)
                frame.image, transform = stabilizer.stabilize(frame.image)

            results = self.detector.detect(frame)
            detections = parse_results(frame.frame_id, results)
            if transform is not None:
                unwarp_boxes(detections, transform)
            all_detections.extend(detections)

        reader.close()

        CsvExporter.export_detections(all_detections, output_csv)

        print(f"\nDetection finished. Detections: {len(all_detections)}")

        if postprocess:
            self._postprocess(output_csv)

    def _postprocess(self, output_csv: Path) -> None:
        """Field filter -> stitch -> track filter -> evaluate, each optional."""

        if self.field_filter and self.field_calibration:
            calibration = PitchCalibration.load(Path(self.field_calibration))
            filter_to_pitch_csv(output_csv, output_csv, calibration, self.field_margin)
        elif self.field_filter and self.field_region:
            region = FieldRegion.load(Path(self.field_region))
            filter_detections_csv(output_csv, output_csv, region)

        if self.stitch:
            stitch_csv(
                output_csv,
                output_csv,
                max_frame_gap=self.stitch_max_frame_gap,
                max_distance_px=self.stitch_max_distance_px,
            )

        if self.track_filter:
            filter_tracks_csv(
                output_csv,
                output_csv,
                pitch_margin=self.track_filter_pitch_margin,
                min_inside_ratio=self.track_filter_min_inside_ratio,
            )

        if self.evaluate:
            evaluate_tracking(output_csv).print_summary()
