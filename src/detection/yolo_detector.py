"""
YOLO object detector.
"""

from ultralytics import YOLO

from src.core.frame import Frame


class YoloDetector:
    """Wrapper around the Ultralytics YOLO model."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        nms_iou: float = 0.7,
        imgsz: int = 640,
        confidence: float = 0.25,
    ) -> None:
        print("Loading YOLO model...")
        self.model = YOLO(model_name)
        self.tracker = tracker
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.confidence = confidence
        print("YOLO loaded.")

    def detect(self, frame: Frame):

        """
        Detect and track objects in a frame.
        """

        return self.model.track(
            frame.image,
            persist=True,
            tracker=self.tracker,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=False,
        )
