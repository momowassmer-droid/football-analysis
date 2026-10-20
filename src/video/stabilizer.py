"""
Camera-motion frame stabilization for the tracking stage.

Veo-style footage pans/zooms a virtual camera. ByteTrack associates boxes
across frames with a constant-velocity Kalman model in image space; a global
camera pan moves every box at once, breaking that assumption and causing ID
switches when players cross. Warping each frame back onto a fixed reference
cancels the global motion, so only true player motion remains and frame-to-
frame association is far more stable.

The detector runs on the stabilized frame, so its boxes come out in reference
coordinates; ``unwarp_boxes`` maps them back to the original frame's pixels so
the rest of the pipeline (colour sampling, rendering, homography) is unchanged
-- the stabilization benefits the tracker only and leaves the stored data in
the original image space.

This alignment is used only to steady the tracker: the detector runs on the
warped frame and ``unwarp_boxes`` puts its boxes back into original pixels, so
nothing downstream sees the warp. The pitch-coordinate mapping and the line
overlay get their own, more accurate per-frame transform from
``src.motion.camera_motion`` (a full projective homography, not this 4-DOF
affine), so the two concerns are kept separate.
"""

import cv2
import numpy as np

from src.core.detection import Detection

MIN_MATCHES = 12
MAX_MATCHES_USED = 200


class ReferenceAligner:
    """Estimates the 2x3 affine mapping a frame onto a fixed reference frame."""

    def __init__(self, reference_bgr: np.ndarray, max_features: int = 1500):
        self.orb = cv2.ORB_create(max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, None)

    def affine_to_reference(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """2x3 affine mapping this frame's pixels onto the reference; None on failure."""
        if self.ref_des is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return None
        matches = self.bf.match(des, self.ref_des)
        if len(matches) < MIN_MATCHES:
            return None
        matches = sorted(matches, key=lambda m: m.distance)[:MAX_MATCHES_USED]
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        return M


class FrameStabilizer:
    """Warps each frame onto a fixed reference frame to cancel camera motion."""

    def __init__(self, reference_bgr: np.ndarray, max_features: int = 1500):
        self.aligner = ReferenceAligner(reference_bgr, max_features)
        self.size = (reference_bgr.shape[1], reference_bgr.shape[0])  # (w, h)

    def stabilize(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Return ``(warped_to_reference, M)`` where ``M`` is the image->reference
        affine (2x3). If estimation fails, the original frame is returned with
        ``M = None`` so the caller can fall back gracefully (boxes left as-is).
        """
        M = self.aligner.affine_to_reference(frame_bgr)
        if M is None:
            return frame_bgr, None
        warped = cv2.warpAffine(frame_bgr, M, self.size)
        return warped, M


def unwarp_boxes(detections: list[Detection], M: np.ndarray | None) -> None:
    """
    Map box coords from reference space back to the original frame, in place.

    ``M`` is the 2x3 image->reference affine the detector's frame was warped by;
    its inverse takes reference coordinates back to the original frame. Each box's
    four corners are mapped and re-bounded to stay axis-aligned (``M`` is a
    similarity, so the re-bounded box stays tight). A ``None`` transform (failed
    estimation -> frame was not warped) leaves the boxes untouched.
    """
    if M is None or not detections:
        return
    inv = cv2.invertAffineTransform(M)
    for d in detections:
        corners = np.float32(
            [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]]
        ).reshape(-1, 1, 2)
        mapped = cv2.transform(corners, inv).reshape(-1, 2)
        d.x1 = float(mapped[:, 0].min())
        d.x2 = float(mapped[:, 0].max())
        d.y1 = float(mapped[:, 1].min())
        d.y2 = float(mapped[:, 1].max())
