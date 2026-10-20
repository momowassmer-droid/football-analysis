"""
Entry point: run the analysis pipeline from a config file.

Usage:
    python -m src.main [path/to/config.yaml]
"""

import sys
from pathlib import Path

from src.config.config_loader import load_config
from src.pipeline.analysis_pipeline import AnalysisPipeline

DEFAULT_CONFIG = Path("src/config/config.yaml")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG

    config = load_config(config_path)

    pipeline = AnalysisPipeline.from_config(config)
    pipeline.run(config.video_input, config.detections_output)


if __name__ == "__main__":
    main()
