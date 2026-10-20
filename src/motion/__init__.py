"""
Camera-motion estimation, reusable across tracking programs.

Turns a clip into a per-frame transform onto a fixed reference frame, so any
tracker can undo pan/zoom by importing the result. See ``camera_motion``.
"""

from src.motion.camera_motion import (
    CameraMotionEstimator,
    CameraState,
    MotionModel,
    MotionTrack,
    PairwiseEstimator,
    camera_motion_for,
    decompose,
)

__all__ = [
    "CameraMotionEstimator",
    "CameraState",
    "MotionModel",
    "MotionTrack",
    "PairwiseEstimator",
    "camera_motion_for",
    "decompose",
]
