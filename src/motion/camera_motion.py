"""
Camera-motion estimation for panning / zooming footage.

Standalone and tracking-agnostic: it turns a video into a per-frame transform
that maps that frame's pixels back onto a fixed reference frame, so *any*
tracking stage (pitch calibration, box stabilization, ball tracking, an
overlay renderer, ...) can undo the camera's pan/zoom by importing the result
instead of each re-deriving it. Nothing here depends on football or on the
calibration code.

Why this replaces the older ``motion_log`` / ``ReferenceAligner`` approach:

  * That one matched *every* frame directly against frame 0 with a 4-DOF
    similarity (``estimateAffinePartial2D``). As the camera pans away from
    frame 0 the shared field-of-view shrinks, matches get sparse and biased,
    and the fit absorbs that as a spurious scale change -- on a real clip an
    apparent ~9% zoom appeared by the end on a camera that never zoomed.
  * A virtual pan/tilt/zoom over a (near-)planar scene is a *projective*
    motion, not a similarity, so forcing a similarity injects error even when
    matching succeeds.

This module instead:
  * estimates a full **homography** per frame pair (falls back to affine, then
    similarity, when a pair is too feature-poor for a stable 8-DOF fit),
  * chains **frame-to-frame** (consecutive frames overlap almost completely,
    which is the easy, reliable case) through a **keyframe** so drift can't
    accumulate arbitrarily: a new keyframe is anchored only when the camera
    has moved far enough that matching against the old one degrades, so the
    transform chain has a handful of links, not hundreds,
  * exposes a decomposition into pan / zoom / rotation for callers that want
    interpretable camera motion rather than a raw matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src.video.video_reader import VideoReader

IDENTITY_H = np.eye(3, dtype=np.float64)


class MotionModel(str, Enum):
    """Geometric model fitted between two frames, cheapest-DOF first."""
    SIMILARITY = "similarity"   # 4 DOF: translation, rotation, uniform scale
    AFFINE = "affine"           # 6 DOF: + shear / non-uniform scale
    HOMOGRAPHY = "homography"   # 8 DOF: full projective (pan/tilt on a plane)


@dataclass(slots=True)
class CameraState:
    """Interpretable camera motion of a frame relative to the reference."""
    dx: float          # horizontal pan in reference pixels
    dy: float          # vertical pan in reference pixels
    zoom: float        # scale factor (1.0 = same, >1 zoomed in)
    rotation_deg: float


def decompose(transform: np.ndarray, at: tuple[float, float] = (0.0, 0.0)) -> CameraState:
    """
    Pan / zoom / rotation of a frame->reference transform, evaluated at ``at``.

    A homography's zoom and rotation are *local* -- they vary across the image
    under perspective -- so reading them off the raw 2x2 block (which also
    carries the projective terms) is wrong and was over-reporting zoom. Instead
    we take the transform's Jacobian at ``at`` (pass the image centre for the
    representative value): that linearisation is the true local scale/rotation
    there, and the pan is how far that point actually moves. Interpretation
    only -- the full matrix is what gets applied.
    """
    x, y = at
    h = transform
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(w) < 1e-12:
        return CameraState(0.0, 0.0, 1.0, 0.0)
    u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    # Jacobian of (u/w, v/w) w.r.t. (x, y) -- the local affine at this point.
    dudx = (h[0, 0] * w - u * h[2, 0]) / w**2
    dudy = (h[0, 1] * w - u * h[2, 1]) / w**2
    dvdx = (h[1, 0] * w - v * h[2, 0]) / w**2
    dvdy = (h[1, 1] * w - v * h[2, 1]) / w**2
    jac = np.array([[dudx, dudy], [dvdx, dvdy]])
    zoom = float(np.sqrt(abs(np.linalg.det(jac))))
    rotation = float(np.degrees(np.arctan2(dvdx, dudx)))
    return CameraState(
        dx=float(u / w - x),
        dy=float(v / w - y),
        zoom=zoom,
        rotation_deg=rotation,
    )


def _to_homography(m: np.ndarray | None) -> np.ndarray | None:
    """Promote a 2x3 affine to a 3x3 homography; pass 3x3 through."""
    if m is None:
        return None
    if m.shape == (3, 3):
        return m.astype(np.float64)
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = m
    return h


def _is_sane(h: np.ndarray, max_zoom: float = 3.0) -> bool:
    """Reject degenerate / wildly-scaled fits that RANSAC occasionally returns."""
    if not np.all(np.isfinite(h)):
        return False
    det = np.linalg.det(h[:2, :2])
    if det <= 1e-6:
        return False
    zoom = np.sqrt(abs(det))
    return 1.0 / max_zoom <= zoom <= max_zoom


@dataclass(slots=True)
class _Features:
    keypoints: tuple
    descriptors: np.ndarray | None


def ground_mask(frame_bgr: np.ndarray, work_width: int = 480) -> np.ndarray:
    """
    Mask (255 = usable) selecting the playing surface, for feature extraction.

    Features must sit on *static* scene content, or the motion estimate follows
    whatever moves: wind-blown tree foliage biases it coherently for seconds at
    a time (no temporal filter can remove that), and spectators/benches add
    jitter. The pitch itself is the one guaranteed-static, always-visible
    surface, so features are restricted to the dominant green region: HSV green
    -> morphological open/close -> largest connected component -> slight
    dilation (so the touchline paint and immediate boundary are included --
    they are the most feature-rich static structures in the scene).

    Computed at ``work_width`` resolution, not full HD: a region mask needs no
    pixel precision, and morphology cost scales with area x kernel-area, so
    1/4 width is ~16x cheaper for an indistinguishable mask (measured: this
    was 43% of the whole per-frame budget at full resolution).
    """
    h_full, w_full = frame_bgr.shape[:2]
    scale = min(1.0, work_width / w_full)
    small = (cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
             if scale < 1.0 else frame_bgr)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    mask = (((h >= 32) & (h <= 95) & (s > 35)).astype(np.uint8)) * 255
    # kernel sizes are the full-res ones (21/25/31) scaled to the work size
    def k(size_full: int) -> int:
        return max(3, int(round(size_full * scale)) | 1)   # odd, >= 3
    # open: sever thin links to (equally green) tree foliage; close: heal the
    # holes players and painted lines punch into the turf region
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k(21), k(21))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k(25), k(25)), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((k(31), k(31)), np.uint8))
    if scale < 1.0:
        mask = cv2.resize(mask, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    return mask


class PairwiseEstimator:
    """Estimates the transform between two frames from ORB feature matches."""

    def __init__(
        self,
        model: MotionModel = MotionModel.HOMOGRAPHY,
        max_features: int = 2000,
        ratio: float = 0.75,
        ransac_thresh: float = 3.0,
        min_inliers: int = 25,
        work_scale: float = 0.5,
    ):
        self.model = model
        self.orb = cv2.ORB_create(max_features)
        # No crossCheck: knnMatch + Lowe ratio test is a stricter match filter,
        # which matters more than crossCheck for a clean fundamental fit.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        # Detect on a downscaled frame; keypoint coordinates are scaled back so
        # every transform stays in full-resolution pixels. Half resolution cuts
        # ORB cost ~3x; the ~1px localisation loss disappears in the RANSAC
        # consensus over hundreds of matches (verified: overlay quality is
        # unchanged on all three test cameras).
        self.work_scale = work_scale

    def features(self, frame_bgr: np.ndarray, mask: np.ndarray | None = None) -> _Features:
        """
        ORB features in full-resolution coordinates.

        ``mask`` is the (full-res) ground mask; pass a cached one to skip the
        per-frame mask cost -- the camera moves slowly, so a mask a few frames
        old still covers the pitch. Restricting to the playing surface keeps
        moving foliage/spectators from biasing the fit (see ground_mask); if
        the mask leaves too little texture, fall back to the full frame.
        """
        if mask is None:
            mask = ground_mask(frame_bgr)
        s = self.work_scale
        if s < 1.0:
            frame_small = cv2.resize(frame_bgr, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(mask, (frame_small.shape[1], frame_small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            frame_small, mask_small = frame_bgr, mask
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, mask_small)
        if des is None or len(des) < 2 * self.min_inliers:
            kp, des = self.orb.detectAndCompute(gray, None)
        if s < 1.0 and kp:
            for keypoint in kp:
                keypoint.pt = (keypoint.pt[0] / s, keypoint.pt[1] / s)
        return _Features(kp, des)

    def _matched_points(
        self, src: _Features, dst: _Features
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if src.descriptors is None or dst.descriptors is None:
            return None
        if len(src.descriptors) < 2 or len(dst.descriptors) < 2:
            return None
        knn = self.bf.knnMatch(src.descriptors, dst.descriptors, k=2)
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            return None
        src_pts = np.float32([src.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([dst.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return src_pts, dst_pts

    def estimate(
        self, src: _Features, dst: _Features
    ) -> tuple[np.ndarray, int] | None:
        """
        Transform mapping ``src`` pixels onto ``dst`` pixels, with inlier count.

        Tries the configured model, then degrades (homography -> affine ->
        similarity) if the richer model finds too few inliers or a degenerate
        fit -- a feature-poor pair is better described by fewer parameters than
        overfitted by more.
        """
        pts = self._matched_points(src, dst)
        if pts is None:
            return None
        src_pts, dst_pts = pts

        order = {
            MotionModel.HOMOGRAPHY: [MotionModel.HOMOGRAPHY, MotionModel.AFFINE, MotionModel.SIMILARITY],
            MotionModel.AFFINE: [MotionModel.AFFINE, MotionModel.SIMILARITY],
            MotionModel.SIMILARITY: [MotionModel.SIMILARITY],
        }[self.model]

        for model in order:
            h, inliers = self._fit(model, src_pts, dst_pts)
            if h is not None and inliers >= self.min_inliers and _is_sane(h):
                return h, inliers
        return None

    def _fit(
        self, model: MotionModel, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> tuple[np.ndarray | None, int]:
        if model is MotionModel.HOMOGRAPHY:
            h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            return _to_homography(h), int(mask.sum()) if mask is not None else 0
        full = model is MotionModel.AFFINE
        estimator = cv2.estimateAffine2D if full else cv2.estimateAffinePartial2D
        m, mask = estimator(src_pts, dst_pts, method=cv2.RANSAC,
                            ransacReprojThreshold=self.ransac_thresh)
        return _to_homography(m), int(mask.sum()) if mask is not None else 0


@dataclass(slots=True)
class MotionTrack:
    """
    Per-frame transform mapping each frame's pixels onto the reference frame.

    ``transforms[i]`` is a 3x3 homography H such that ``x_reference ~ H @ x_i``.
    The reference frame maps with identity. Reusable across tracking programs:
    save once, load anywhere.
    """
    transforms: dict[int, np.ndarray]
    reference_frame_id: int
    model: MotionModel
    frame_width: int
    frame_height: int
    anchored_keyframes: list[int]   # frames promoted to keyframes (diagnostic)
    hold_frames: list[int]          # frames where estimation failed -> held previous

    def to_reference(self, frame_id: int) -> np.ndarray:
        return self.transforms.get(frame_id, IDENTITY_H)

    def median_displacement(self) -> float:
        """
        Median over all frames of how far the image-centre point moves (px)
        -- "how much does this camera actually pan". Measured at the centre,
        not the corners: corner displacement amplifies the zoom/rotation
        *noise* of a static camera (0.5-1% spurious zoom = tens of px at the
        corners but almost nothing at the centre), which would make a static
        camera look like a moving one to the significance gate.
        """
        w, h = float(self.frame_width), float(self.frame_height)
        centre = np.array([[[w / 2, h / 2]]], dtype=np.float32)
        per_frame = []
        for t in self.transforms.values():
            moved = cv2.perspectiveTransform(centre, t).reshape(2)
            per_frame.append(float(np.hypot(moved[0] - w / 2, moved[1] - h / 2)))
        return float(np.median(per_frame)) if per_frame else 0.0

    def interpolate_outliers(self, max_gap: int = 5, threshold_px: float = 12.0) -> "MotionTrack":
        """
        Replace short spikes in the track by temporal interpolation.

        A frame is an outlier when its image-centre position jumps by more
        than ``threshold_px`` against the local median trajectory (window of
        11) -- a real pan moves the whole neighbourhood with it, so it never
        trips this; a single-frame mis-estimate does. Runs of at most
        ``max_gap`` consecutive outliers are replaced by linearly blending
        the transforms of the nearest good frames on either side; longer runs
        are left alone (that is sustained motion, not a spike).
        """
        frames = sorted(self.transforms)
        if len(frames) < 5:
            return self
        w, h = float(self.frame_width), float(self.frame_height)
        centre = np.array([[[w / 2, h / 2]]], dtype=np.float32)
        pos = np.array([
            cv2.perspectiveTransform(centre, self.transforms[f]).reshape(2)
            for f in frames
        ])
        half = 5
        is_outlier = np.zeros(len(frames), dtype=bool)
        for i in range(len(frames)):
            lo, hi = max(0, i - half), min(len(frames), i + half + 1)
            local = np.median(pos[lo:hi], axis=0)
            if np.hypot(*(pos[i] - local)) > threshold_px:
                is_outlier[i] = True

        out = {f: self.transforms[f].copy() for f in frames}
        i = 0
        while i < len(frames):
            if not is_outlier[i]:
                i += 1
                continue
            j = i
            while j < len(frames) and is_outlier[j]:
                j += 1
            run = j - i
            if run <= max_gap and i > 0 and j < len(frames):
                a = self.transforms[frames[i - 1]]
                b = self.transforms[frames[j]]
                a = a / a[2, 2]
                b = b / b[2, 2]
                for k in range(i, j):
                    t = (k - (i - 1)) / (j - (i - 1))
                    out[frames[k]] = (1.0 - t) * a + t * b
            i = j

        return MotionTrack(
            transforms=out,
            reference_frame_id=self.reference_frame_id,
            model=self.model,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            anchored_keyframes=list(self.anchored_keyframes),
            hold_frames=list(self.hold_frames),
        )

    def smoothed(self, window: int = 5) -> "MotionTrack":
        """
        Median-filter the per-frame transforms over a temporal window.

        A virtual (Veo-style) camera moves smoothly, but per-frame feature
        estimates jitter when features sit on non-static content -- wind-blown
        foliage, people at the bench -- producing single-frame spikes (observed:
        zoom bouncing 0.98 -> 1.06 -> 0.96 within ten frames on a camera that
        barely moved, tilting the whole overlay on exactly those frames). An
        element-wise temporal median removes such spikes without lagging a real
        pan: the median tracks any motion sustained longer than half the window
        and ignores anything shorter. Transforms are normalised (h33 = 1)
        before filtering so their elements are comparable across frames.
        """
        frames = sorted(self.transforms)
        stack = np.stack([
            self.transforms[f] / self.transforms[f][2, 2] for f in frames
        ])
        half = max(1, window // 2)
        out: dict[int, np.ndarray] = {}
        for i, f in enumerate(frames):
            lo, hi = max(0, i - half), min(len(frames), i + half + 1)
            out[f] = np.median(stack[lo:hi], axis=0)
        return MotionTrack(
            transforms=out,
            reference_frame_id=self.reference_frame_id,
            model=self.model,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            anchored_keyframes=list(self.anchored_keyframes),
            hold_frames=list(self.hold_frames),
        )

    def camera_state(self, frame_id: int) -> CameraState:
        """Pan/zoom/rotation at the image centre (the representative point)."""
        centre = (self.frame_width / 2.0, self.frame_height / 2.0)
        return decompose(self.to_reference(frame_id), at=centre)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = np.array(sorted(self.transforms), dtype=np.int64)
        stacked = np.stack([self.transforms[int(f)] for f in frames])
        np.savez(
            path,
            frames=frames,
            transforms=stacked,
            reference_frame_id=self.reference_frame_id,
            model=self.model.value,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            anchored_keyframes=np.array(self.anchored_keyframes, dtype=np.int64),
            hold_frames=np.array(self.hold_frames, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> "MotionTrack":
        data = np.load(path, allow_pickle=False)
        transforms = {int(f): data["transforms"][i] for i, f in enumerate(data["frames"])}
        return cls(
            transforms=transforms,
            reference_frame_id=int(data["reference_frame_id"]),
            model=MotionModel(str(data["model"])),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            anchored_keyframes=[int(x) for x in data["anchored_keyframes"]],
            hold_frames=[int(x) for x in data["hold_frames"]],
        )


class _PhotometricVerifier:
    """
    Per-frame A/B test of candidate transforms against the actual pixels.

    Feature-based estimates on a near-static camera carry ~0.5-1.5% spurious
    zoom / a few px of pan; applying such an estimate moves the pitch mapping
    *away* from the truth. Magnitude cannot tell that noise from a genuinely
    small pan (measured: ~11px centre displacement was real motion on one
    clip and pure noise on another), so the decision is made on image
    evidence instead: warp the (downscaled, ground-masked) frame onto the
    reference with each candidate transform and keep whichever leaves the
    smallest median grey-level difference. Players and shadows corrupt only a
    minority of masked pixels, which the median ignores; downscaling keeps
    the check cheap (one small warp per candidate).
    """

    def __init__(self, reference_bgr: np.ndarray, scale: float = 0.25,
                 erode_px: int = 12, edge_quantile: float = 0.90):
        small = cv2.resize(reference_bgr, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.ref_gray = gray.astype(np.int16)
        h, w = gray.shape
        mask = cv2.resize(ground_mask(reference_bgr), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        # Evaluate only at strong-gradient pixels well INSIDE the ground mask:
        # flat turf is insensitive to alignment (it dominated a plain median
        # and made the check blind), and the mask fringe carries fence/kerb
        # edges whose alignment can trade off against the pitch lines. What
        # remains after eroding the mask and keeping the top gradient decile
        # is essentially the painted lines -- exactly the structure whose
        # alignment the whole system is judged by.
        interior = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8)) > 0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude = np.hypot(gx, gy)
        if interior.any():
            threshold = np.quantile(magnitude[interior], edge_quantile)
            self.ref_edges = (magnitude >= threshold) & interior
        else:
            self.ref_edges = interior
        self.scale_to_small = np.diag([scale, scale, 1.0])
        self.scale_to_full = np.diag([1.0 / scale, 1.0 / scale, 1.0])

    def downscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = self.ref_gray.shape
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def error(self, gray_small: np.ndarray, transform: np.ndarray) -> float:
        """Median grey difference at the reference's interior line edges."""
        h, w = self.ref_gray.shape
        h_small = self.scale_to_small @ transform @ self.scale_to_full
        warped = cv2.warpPerspective(gray_small, h_small, (w, h))
        valid = cv2.warpPerspective(
            np.full(gray_small.shape, 255, np.uint8), h_small, (w, h)) > 128
        usable = self.ref_edges & valid
        if int(usable.sum()) < 100:
            return float("inf")
        diff = np.abs(warped.astype(np.int16) - self.ref_gray)[usable]
        return float(np.median(diff))

    def best(self, gray_small: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray:
        """The candidate transform the pixels support best."""
        errors = [self.error(gray_small, c) for c in candidates]
        return candidates[int(np.argmin(errors))]


class CameraMotionEstimator:
    """
    Builds a :class:`MotionTrack`, preferring a direct match to the reference.

    Matching every frame straight to the reference frame is the zero-drift
    ideal: no error accumulates because nothing is chained. It only breaks when
    the camera has panned so far that the reference and current frame barely
    overlap. So each frame is first matched directly to the reference; only if
    that match is too weak do we fall back to a *single* hop from the most
    recent frame that *did* anchor to the reference (the "keyframe"). Because
    the fallback is always one hop from a reference-anchored frame -- never a
    hop off another hop -- drift stays bounded even during a long excursion,
    and direct (zero-drift) matching resumes automatically as the camera pans
    back. (An earlier version that re-anchored greedily whenever the reference
    match merely *weakened* chained through dozens of keyframes and measurably
    drifted; matching to the reference is preferred here precisely to avoid
    that.)
    """

    def __init__(
        self,
        estimator: PairwiseEstimator | None = None,
        min_reference_inliers: int = 40,
        smooth_window: int = 5,
        mask_every: int = 15,
        frame_stride: int = 2,
    ):
        self.estimator = estimator or PairwiseEstimator()
        self.min_reference_inliers = min_reference_inliers
        # Temporal median over the final track (see MotionTrack.smoothed);
        # <= 1 disables. On by default because single-frame estimation spikes
        # (moving foliage / people grabbing the feature budget) otherwise tilt
        # the overlay and the pitch mapping on exactly those frames.
        self.smooth_window = smooth_window
        # Recompute the ground mask only every N frames: the camera moves a
        # few px per frame, so a slightly stale region mask is still correct,
        # and the mask was a large share of the per-frame cost.
        self.mask_every = max(1, mask_every)
        # Estimate every Nth frame and interpolate the ones between: a virtual
        # camera's motion over 1-2 frames is far below the estimator's own
        # noise floor, so the interpolation error is invisible while the whole
        # pipeline gets ~stride times faster.
        self.frame_stride = max(1, frame_stride)
        # NOTE: an earlier version had a whole-clip "static camera" gate here
        # (median motion below a threshold -> force identity). It misfired on
        # a mostly-still camera that pans occasionally: the median is tiny, so
        # the gate killed compensation exactly on the panning frames. The
        # root cause of the noise it tried to suppress (features on moving
        # foliage) is fixed at the source now via ``ground_mask`` plus the
        # per-frame photometric verification.

    def track(self, video_path: Path) -> MotionTrack:
        reader = VideoReader(video_path)
        reader.open()

        transforms: dict[int, np.ndarray] = {}
        anchored: list[int] = []
        holds: list[int] = []
        frame_w = frame_h = 0

        reference_id: int | None = None
        ref_features: _Features | None = None
        verifier: _PhotometricVerifier | None = None

        # Fallback anchor: the most recent frame that matched the reference
        # directly, and its (drift-free) transform to the reference.
        key_features: _Features | None = None
        key_to_ref = IDENTITY_H

        prev_features: _Features | None = None
        prev_to_ref = IDENTITY_H

        mask: np.ndarray | None = None
        frames_seen = 0
        skipped: list[int] = []

        for frame in reader.frames():
            if frame_w == 0:
                frame_h, frame_w = frame.image.shape[:2]

            if reference_id is not None and (frames_seen % self.frame_stride) != 0:
                # interpolated later, between the surrounding estimated frames
                skipped.append(frame.frame_id)
                frames_seen += 1
                continue
            frames_seen += 1

            if mask is None or (frames_seen - 1) % self.mask_every == 0:
                mask = ground_mask(frame.image)
            feats = self.estimator.features(frame.image, mask)

            if reference_id is None:                     # first frame = reference
                reference_id = frame.frame_id
                ref_features = key_features = feats
                verifier = _PhotometricVerifier(frame.image)
                transforms[frame.frame_id] = IDENTITY_H
                prev_features, prev_to_ref = feats, IDENTITY_H
                continue

            direct = self.estimator.estimate(feats, ref_features)
            if direct is not None and direct[1] >= self.min_reference_inliers:
                cur_to_ref = direct[0]                   # zero-drift direct match
                key_features, key_to_ref = feats, cur_to_ref   # newest anchor
            else:
                # Reference too far: one hop from the last reference-anchored frame.
                anchored.append(frame.frame_id)
                hop = self.estimator.estimate(feats, key_features)
                if hop is not None:
                    cur_to_ref = key_to_ref @ hop[0]
                else:
                    hop = self.estimator.estimate(feats, prev_features)
                    if hop is not None:
                        cur_to_ref = prev_to_ref @ hop[0]
                    else:
                        cur_to_ref = prev_to_ref         # hold: estimation failed
                        holds.append(frame.frame_id)

            # Let the pixels arbitrate: the feature estimate competes against
            # "camera did not move" (identity) and "same as last frame". This
            # catches the residual noise of near-static frames that no
            # magnitude threshold can separate from a genuinely small pan.
            cur_to_ref = verifier.best(
                verifier.downscale(frame.image),
                [cur_to_ref, IDENTITY_H, prev_to_ref],
            )

            transforms[frame.frame_id] = cur_to_ref
            prev_features, prev_to_ref = feats, cur_to_ref

        reader.close()

        # Fill the frames skipped by the stride: linear interpolation between
        # the two surrounding estimated transforms (normalised; the per-frame
        # deltas are tiny, so linear blending of the matrices is exact enough).
        estimated = np.array(sorted(transforms), dtype=np.int64)
        for f in skipped:
            idx = int(np.searchsorted(estimated, f))
            lo = int(estimated[idx - 1]) if idx > 0 else None
            hi = int(estimated[idx]) if idx < len(estimated) else None
            if lo is None or hi is None:
                anchor = lo if lo is not None else hi
                transforms[f] = transforms[anchor] if anchor is not None else IDENTITY_H
                continue
            t = (f - lo) / (hi - lo)
            a = transforms[lo] / transforms[lo][2, 2]
            b = transforms[hi] / transforms[hi][2, 2]
            transforms[f] = (1.0 - t) * a + t * b

        track = MotionTrack(
            transforms=transforms,
            reference_frame_id=reference_id if reference_id is not None else 0,
            model=self.estimator.model,
            frame_width=frame_w,
            frame_height=frame_h,
            anchored_keyframes=anchored,
            hold_frames=holds,
        )
        track = track.interpolate_outliers()
        if self.smooth_window > 1:
            track = track.smoothed(self.smooth_window)
        return track


def camera_motion_for(
    video_path: Path,
    cache_path: Path | None = None,
    estimator: CameraMotionEstimator | None = None,
) -> MotionTrack:
    """Load a cached motion track, or compute and cache one if absent."""
    if cache_path is not None and cache_path.exists():
        return MotionTrack.load(cache_path)
    track = (estimator or CameraMotionEstimator()).track(video_path)
    if cache_path is not None:
        track.save(cache_path)
    return track
