"""Run reproducible tracking experiments on the short validation clip."""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from src.config.config_loader import load_config
from src.evaluation.tracking_metrics import evaluate_tracking
from src.pipeline.analysis_pipeline import AnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("src/config/config.yaml"))
    parser.add_argument("--benchmark-config", type=Path, default=Path("config/tracking_benchmark.yaml"))
    parser.add_argument("--only", nargs="*", help="Run only these experiment names")
    args = parser.parse_args()

    with args.benchmark_config.open("r", encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)

    base = load_config(args.base_config)
    video_path = Path(benchmark["video"])
    output_dir = Path(benchmark["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [])
    summaries: list[dict[str, object]] = []

    for experiment in benchmark["experiments"]:
        name = experiment["name"]
        if selected and name not in selected:
            continue

        output_csv = output_dir / f"{name}.csv"
        detection = replace(
            base.detection,
            model=experiment.get("model", base.detection.model),
            imgsz=experiment.get("imgsz", base.detection.imgsz),
            confidence=experiment.get("confidence", base.detection.confidence),
            nms_iou=experiment.get("nms_iou", base.detection.nms_iou),
        )
        tracking = replace(
            base.tracking,
            tracker=experiment.get("tracker", base.tracking.tracker),
        )
        config = replace(
            base,
            video_input=video_path,
            detections_output=output_csv,
            detection=detection,
            tracking=tracking,
            evaluate=False,
        )

        print(f"\n===== EXPERIMENT: {name} =====")
        AnalysisPipeline.from_config(config).run(video_path, output_csv)
        report = evaluate_tracking(output_csv)
        report.print_summary()

        summaries.append(
            {
                "experiment": name,
                "model": detection.model,
                "tracker": tracking.tracker,
                "imgsz": detection.imgsz,
                "confidence": detection.confidence,
                "nms_iou": detection.nms_iou,
                "detections": report.total_detections,
                "frames": report.total_frames,
                "unique_tracks": report.unique_track_ids,
                "mean_track_length": report.mean_track_length,
                "median_track_length": report.median_track_length,
                "short_tracks_ratio": report.short_tracks_ratio,
                "mean_detections_per_frame": report.mean_detections_per_frame,
                "suspected_id_switches": report.suspected_id_switches,
                "fragmentation_links": report.fragmentation_links,
            }
        )

    if not summaries:
        raise SystemExit("No matching experiments selected.")

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nWrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
