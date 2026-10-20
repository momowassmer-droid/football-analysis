"""
Hebel 2: low-confidence ball detection.

The pipeline's normal detection pass uses a confidence threshold tuned for
players, which discards many faint ball detections (the ball is small and
motion-blurred). This pass re-runs the detector over the video for the ball
class only at a much lower threshold, accepting more candidates. The
single-ball physics in the Kalman tracker then gates away the false ones,
so low precision here is fine -- recall is what matters.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from src.video.video_reader import VideoReader

# COCO class ids the ball gets detected (and mis-detected) as.
BALL_COCO_IDS = [29, 32]  # 29 = frisbee, 32 = sports ball


def detect_ball_lowconf(
    video_path: Path,
    output_csv: Path,
    model_name: str = "yolov8n.pt",
    conf: float = 0.10,
    imgsz: int = 1920,
) -> pd.DataFrame:

    model = YOLO(model_name)
    reader = VideoReader(video_path)
    reader.open()

    rows = []
    for frame in tqdm(reader.frames(), total=reader.metadata.frame_count,
                      desc="Ball low-conf", unit="frame"):
        result = model.predict(
            frame.image, conf=conf, imgsz=imgsz,
            classes=BALL_COCO_IDS, verbose=False,
        )[0]
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "frame": frame.frame_id,
                "track_id": None,
                "class_id": int(box.cls),
                "class_name": names[int(box.cls)],
                "confidence": float(box.conf),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    reader.close()

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Low-conf ball: {len(df)} candidates over {reader.metadata.frame_count} frames "
          f"(conf>={conf})")
    return df
