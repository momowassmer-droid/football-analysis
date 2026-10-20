"""
Core data model for video metadata.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class VideoMetadata:
    """
    Metadata describing a video file.
    """

    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str | None = None