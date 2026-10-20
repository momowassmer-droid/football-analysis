"""
Convert YOLO results into Detection objects.
"""

from src.core.detection import Detection


def parse_results(frame_id: int, results) -> list[Detection]:

    detections: list[Detection] = []

    result = results[0]

    names = result.names

    for box in result.boxes:

        track_id = None

        if box.id is not None:
            track_id = int(box.id)

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            Detection(
                frame_id=frame_id,
                track_id=track_id,
                class_id=int(box.cls),
                class_name=names[int(box.cls)],
                confidence=float(box.conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return detections