"""
kpe/extract/optical_flow.py

Sparse optical flow extractor using Lucas-Kanade with Shi-Tomasi corner detection.
Produces a 10-dimensional motion descriptor per frame pair.

Layer: Extract (Layer 2 of the KPE 5-layer pipeline)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FlowConfig:
    """
    All tunable parameters for the sparse optical flow extractor.
    Override defaults via constructor — nothing is hard-coded.
    """

    # --- Frame resolution (should match ingest resize) ---
    frame_width: int = 320
    frame_height: int = 240

    # --- Shi-Tomasi corner detection ---
    max_corners: int = 200          # Max feature points to track
    quality_level: float = 0.01    # Min accepted quality ratio (0–1)
    min_distance: float = 7.0      # Min px distance between corners
    block_size: int = 7            # Neighbourhood size for corner detection

    # --- Lucas-Kanade iterative search ---
    lk_win_size: tuple = (15, 15)  # Search window per pyramid level
    lk_max_level: int = 2          # Pyramid levels (0 = no pyramid)
    lk_max_iter: int = 10          # Max iterations per level
    lk_epsilon: float = 0.03       # Convergence epsilon

    # --- Descriptor binning ---
    n_magnitude_bins: int = 4      # Histogram bins for flow magnitude
    n_angle_bins: int = 4          # Histogram bins for flow angle (0–360°)
    magnitude_clip: float = 20.0   # Clip magnitude outliers above this px/frame

    # --- Quality filter ---
    min_tracked_points: int = 10   # Discard frame if fewer points tracked

    # Derived: descriptor dimensionality
    @property
    def descriptor_dim(self) -> int:
        return self.n_magnitude_bins + self.n_angle_bins + 2  # +2 for mean dx, dy

    def lk_criteria(self):
        """Build the cv2 termination criteria tuple from config."""
        return (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.lk_max_iter,
            self.lk_epsilon,
        )

    def shi_tomasi_params(self) -> dict:
        return {
            "maxCorners": self.max_corners,
            "qualityLevel": self.quality_level,
            "minDistance": self.min_distance,
            "blockSize": self.block_size,
        }

    def lk_params(self) -> dict:
        return {
            "winSize": self.lk_win_size,
            "maxLevel": self.lk_max_level,
            "criteria": self.lk_criteria(),
        }


# ---------------------------------------------------------------------------
# Frame descriptor
# ---------------------------------------------------------------------------

@dataclass
class MotionDescriptor:
    """
    10-dimensional motion descriptor for a single frame transition.

    Fields
    ------
    vector      : np.ndarray, shape (descriptor_dim,)
    frame_index : int
    n_points    : int   — number of successfully tracked points
    valid       : bool  — False if tracking quality was too low
    """
    vector: np.ndarray
    frame_index: int
    n_points: int
    valid: bool = True

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "n_points": self.n_points,
            "valid": self.valid,
            "vector": self.vector.tolist(),
        }


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class SparseOpticalFlowExtractor:
    """
    Extracts a stream of MotionDescriptors from sequential video frames
    using sparse Lucas-Kanade optical flow.

    Usage
    -----
    cfg = FlowConfig(max_corners=150, lk_max_level=3)
    extractor = SparseOpticalFlowExtractor(cfg)
    descriptors = extractor.process_video("match_clip.mp4")
    signature = extractor.build_signature(descriptors)  # shape (T, 10)
    """

    def __init__(self, config: Optional[FlowConfig] = None, debug: bool = False):
        self.cfg = config or FlowConfig()
        self.debug = debug
        logger.info(
            "SparseOpticalFlowExtractor initialised — "
            "descriptor_dim=%d, max_corners=%d, pyramid_levels=%d",
            self.cfg.descriptor_dim,
            self.cfg.max_corners,
            self.cfg.lk_max_level,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(self, source) -> list[MotionDescriptor]:
        """
        Extract MotionDescriptors from a video file or cv2 capture object.

        Parameters
        ----------
        source : str | int | cv2.VideoCapture
            File path, camera index, or an already-opened VideoCapture.

        Returns
        -------
        list[MotionDescriptor]
            One descriptor per frame pair. Length = n_frames - 1.
        """
        cap = self._open_capture(source)
        descriptors: list[MotionDescriptor] = []

        try:
            prev_gray = self._read_gray_frame(cap)
            if prev_gray is None:
                logger.warning("Video source produced no frames: %s", source)
                return descriptors

            prev_pts = self._detect_corners(prev_gray)
            frame_idx = 0

            while True:
                curr_gray = self._read_gray_frame(cap)
                if curr_gray is None:
                    break  # End of stream

                frame_idx += 1
                desc = self._compute_descriptor(
                    prev_gray, curr_gray, prev_pts, frame_idx
                )
                descriptors.append(desc)

                # Re-detect corners every frame if tracking failed or thinned out
                if not desc.valid or desc.n_points < self.cfg.min_tracked_points * 2:
                    prev_pts = self._detect_corners(curr_gray)
                else:
                    # Forward-propagate tracked points as new seed
                    prev_pts = self._track_forward(prev_gray, curr_gray, prev_pts)

                prev_gray = curr_gray

        finally:
            if isinstance(source, (str, int)):
                cap.release()

        logger.info(
            "Processed %d frame pairs — %d valid descriptors",
            frame_idx,
            sum(d.valid for d in descriptors),
        )
        return descriptors

    def process_frames(self, frames: list[np.ndarray]) -> list[MotionDescriptor]:
        """
        Extract descriptors from an in-memory list of BGR or grayscale frames.
        Useful for testing without a video file.
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to compute optical flow.")

        descriptors = []
        grays = [self._to_gray(f) for f in frames]
        prev_pts = self._detect_corners(grays[0])

        for i in range(1, len(grays)):
            desc = self._compute_descriptor(grays[i - 1], grays[i], prev_pts, i)
            descriptors.append(desc)
            prev_pts = self._detect_corners(grays[i]) if not desc.valid else \
                       self._track_forward(grays[i - 1], grays[i], prev_pts)

        return descriptors

    def build_signature(
        self,
        descriptors: list[MotionDescriptor],
        fill_invalid: bool = True,
        max_interp_gap: int = 5,
        smooth_window: int = 3,
        l2_normalize: bool = True
    ) -> np.ndarray:
        """
        Stack MotionDescriptors into a 2-D signature array S(t).
        Applies gap interpolation, median smoothing, and L2 normalization.

        Parameters
        ----------
        descriptors  : list[MotionDescriptor]
        fill_invalid : bool
            If True, replace invalid frames with zero vectors (DTW-safe).
            If False, drop invalid frames entirely (shorter but cleaner).
        max_interp_gap : int
            Maximum consecutive invalid frames to linearly interpolate.
        smooth_window : int
            Window size for temporal smoothing via rolling median.
        l2_normalize : bool
            If True, L2 normalize the descriptor of each frame.

        Returns
        -------
        np.ndarray, shape (T, descriptor_dim)
            The S(t) motion signature stream ready for DTW matching.
        """
        if not descriptors:
            raise ValueError("Cannot build signature from empty descriptor list.")

        dim = self.cfg.descriptor_dim
        rows = []
        is_valid = []

        # 1. Base collection
        for d in descriptors:
            if d.valid:
                rows.append(d.vector)
                is_valid.append(True)
            elif fill_invalid:
                rows.append(np.zeros(dim, dtype=np.float32))
                is_valid.append(False)
        
        # If not filling invalid, just stack and optionally normalize
        if not fill_invalid:
            sig = np.stack(rows, axis=0).astype(np.float32)
            if l2_normalize:
                norms = np.linalg.norm(sig, axis=1, keepdims=True)
                sig = np.divide(sig, norms, out=np.zeros_like(sig), where=norms>1e-6)
            return sig

        sig = np.stack(rows, axis=0).astype(np.float32)
        is_valid = np.array(is_valid, dtype=bool)
        n_frames = len(sig)

        # 2. Linear interpolation for small gaps
        if max_interp_gap > 0:
            import itertools
            import operator
            invalid_indices = np.where(~is_valid)[0]
            if len(invalid_indices) > 0 and is_valid.any():
                for k, g in itertools.groupby(enumerate(invalid_indices), lambda ix: ix[0] - ix[1]):
                    group = list(map(operator.itemgetter(1), g))
                    if len(group) < max_interp_gap:
                        start = group[0] - 1
                        end = group[-1] + 1
                        if start >= 0 and end < n_frames and is_valid[start] and is_valid[end]:
                            x = [start, end]
                            for d_idx in range(dim):
                                y = [sig[start, d_idx], sig[end, d_idx]]
                                sig[group, d_idx] = np.interp(group, x, y)
                            is_valid[group] = True

        # 3. Temporal Smoothing (Median Filter)
        if smooth_window > 1:
            pad = smooth_window // 2
            sig_smoothed = np.copy(sig)
            padded_sig = np.pad(sig, ((pad, pad), (0, 0)), mode='edge')
            for i in range(n_frames):
                window = padded_sig[i:i + smooth_window, :]
                sig_smoothed[i] = np.median(window, axis=0)
            sig = sig_smoothed

        # 4. Feature Normalisation (L2)
        if l2_normalize:
            norms = np.linalg.norm(sig, axis=1, keepdims=True)
            sig = np.divide(sig, norms, out=np.zeros_like(sig), where=norms>1e-6)

        logger.info("Built S(t) signature: shape=%s", sig.shape)
        return sig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_descriptor(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: Optional[np.ndarray],
        frame_idx: int,
    ) -> MotionDescriptor:
        """
        Core: track points, compute flow vectors, bin into descriptor.
        """
        zero_vec = np.zeros(self.cfg.descriptor_dim, dtype=np.float32)

        if prev_pts is None or len(prev_pts) == 0:
            return MotionDescriptor(zero_vec, frame_idx, 0, valid=False)

        # --- Lucas-Kanade tracking ---
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            **self.cfg.lk_params(),
        )

        if status is None:
            return MotionDescriptor(zero_vec, frame_idx, 0, valid=False)

        good_mask = status.ravel() == 1
        n_tracked = int(good_mask.sum())

        if n_tracked < self.cfg.min_tracked_points:
            logger.debug(
                "Frame %d: only %d points tracked (min=%d), marking invalid",
                frame_idx, n_tracked, self.cfg.min_tracked_points,
            )
            return MotionDescriptor(zero_vec, frame_idx, n_tracked, valid=False)

        p0 = prev_pts[good_mask].reshape(-1, 2)
        p1 = curr_pts[good_mask].reshape(-1, 2)

        if self.debug:
            vis_frame = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
            for (new_pt, old_pt) in zip(p1, p0):
                a, b = int(new_pt[0]), int(new_pt[1])
                c, d = int(old_pt[0]), int(old_pt[1])
                cv2.line(vis_frame, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(vis_frame, (a, b), 3, (0, 0, 255), -1)
            cv2.imshow("Tracking Debug", vis_frame)
            cv2.waitKey(1)

        # --- Flow vectors ---
        flow = p1 - p0                              # shape (N, 2)
        dx, dy = flow[:, 0], flow[:, 1]

        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle_deg = np.degrees(np.arctan2(dy, dx)) % 360.0

        # Clip magnitude outliers (e.g. jitter artefacts)
        magnitude = np.clip(magnitude, 0, self.cfg.magnitude_clip)

        # --- Build descriptor ---
        mag_hist, _ = np.histogram(
            magnitude,
            bins=self.cfg.n_magnitude_bins,
            range=(0, self.cfg.magnitude_clip),
        )
        ang_hist, _ = np.histogram(
            angle_deg,
            bins=self.cfg.n_angle_bins,
            range=(0.0, 360.0),
        )

        # Normalise histograms to [0, 1]
        mag_hist = mag_hist.astype(np.float32) / (n_tracked + 1e-6)
        ang_hist = ang_hist.astype(np.float32) / (n_tracked + 1e-6)

        # Mean displacement (global motion signal)
        mean_dx = float(np.mean(dx))
        mean_dy = float(np.mean(dy))

        vector = np.concatenate([
            mag_hist,
            ang_hist,
            np.array([mean_dx, mean_dy], dtype=np.float32),
        ]).astype(np.float32)

        assert len(vector) == self.cfg.descriptor_dim, (
            f"Descriptor dim mismatch: got {len(vector)}, "
            f"expected {self.cfg.descriptor_dim}"
        )

        return MotionDescriptor(vector, frame_idx, n_tracked, valid=True)

    def _track_forward(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> np.ndarray:
        """
        Run LK tracking and return only the successfully tracked points
        in the current frame as new seed points.
        """
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.cfg.lk_params()
        )
        if status is None:
            return self._detect_corners(curr_gray)
        good = status.ravel() == 1
        tracked = curr_pts[good]
        if len(tracked) < self.cfg.min_tracked_points:
            return self._detect_corners(curr_gray)
        return tracked.reshape(-1, 1, 2).astype(np.float32)

    def _detect_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corners; returns None if none found."""
        pts = cv2.goodFeaturesToTrack(gray, **self.cfg.shi_tomasi_params())
        if pts is None or len(pts) == 0:
            logger.debug("No corners detected in frame.")
            return None
        return pts.astype(np.float32)

    def _read_gray_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Read one frame and convert to grayscale. Returns None at stream end."""
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return self._to_gray(frame)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR or already-gray frame to grayscale."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _open_capture(self, source) -> cv2.VideoCapture:
        """Accept file path, camera index, or pre-opened VideoCapture."""
        if isinstance(source, cv2.VideoCapture):
            return source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source!r}")
        return cap