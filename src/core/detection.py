"""
Core data model for a single object detection.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    """
    Represents a single detected object in a single video frame.
    """

    frame_id: int
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1
