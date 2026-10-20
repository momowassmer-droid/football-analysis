"""
Core data model for a video frame.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Frame:
    """
    Represents a single frame of a video.
    """

    frame_id: int
    timestamp: float
    image: np.ndarray