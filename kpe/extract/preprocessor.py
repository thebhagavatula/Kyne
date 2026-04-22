"""
kpe/extract/preprocessor.py

Video preprocessor for the KPE Layer 2 (Extract) pipeline.
Handles resolution downscaling and frame skipping optimisation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PreprocessorConfig:
    """Configuration for video preprocessing."""
    target_width: int = 320
    target_height: int = 240
    frame_skip: int = 1
    max_frames: Optional[int] = None

class VideoPreprocessor:
    """
    Preprocesses video sources by resizing, converting to grayscale, and skipping frames.
    """
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.cfg = config or PreprocessorConfig()
        logger.info(
            "VideoPreprocessor initialised - target_res=%dx%d, frame_skip=%d, max_frames=%s",
            self.cfg.target_width,
            self.cfg.target_height,
            self.cfg.frame_skip,
            self.cfg.max_frames
        )

    def preprocess(self, source: Union[str, int, cv2.VideoCapture, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Extracts, resizes, and grayscales frames from a video source with frame skipping.
        Also accepts an in-memory list of frames.
        """
        frames_out = []
        extracted_count = 0
        
        if isinstance(source, list):
            # In-memory frames
            for frame_idx, frame in enumerate(source):
                if frame_idx % self.cfg.frame_skip == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    resized = cv2.resize(gray, (self.cfg.target_width, self.cfg.target_height), interpolation=cv2.INTER_AREA)
                    frames_out.append(resized)
                    extracted_count += 1
                if self.cfg.max_frames and extracted_count >= self.cfg.max_frames:
                    break
            dropped = len(source) - extracted_count
            logger.info("Preprocessed %d in-memory frames. Sampled: %d, Dropped: %d.", len(source), extracted_count, dropped)
            return frames_out
        
        cap = self._open_capture(source)
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                if frame_idx % self.cfg.frame_skip == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    resized = cv2.resize(gray, (self.cfg.target_width, self.cfg.target_height), interpolation=cv2.INTER_AREA)
                    frames_out.append(resized)
                    extracted_count += 1
                    
                frame_idx += 1
                if self.cfg.max_frames and extracted_count >= self.cfg.max_frames:
                    break
        finally:
            if isinstance(source, (str, int)):
                cap.release()
                
        dropped = frame_idx - extracted_count
        logger.info("Preprocessed %d stream frames. Sampled: %d, Dropped: %d.", frame_idx, extracted_count, dropped)
        return frames_out

    def frame_timestamps(self, source: Union[str, int, cv2.VideoCapture, List[np.ndarray]]) -> List[float]:
        """
        Returns the original video timestamps (in seconds) corresponding to each sampled frame.
        If an in-memory list is provided, returns synthetic timestamps assuming 30fps.
        """
        timestamps = []
        extracted_count = 0
        
        if isinstance(source, list):
            # Synthetic timestamps assuming 30 fps
            for frame_idx in range(len(source)):
                if frame_idx % self.cfg.frame_skip == 0:
                    timestamps.append(frame_idx / 30.0)
                    extracted_count += 1
                if self.cfg.max_frames and extracted_count >= self.cfg.max_frames:
                    break
            return timestamps

        cap = self._open_capture(source)
        frame_idx = 0
        
        try:
            while True:
                ret = cap.grab()
                if not ret:
                    break
                    
                if frame_idx % self.cfg.frame_skip == 0:
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    timestamps.append(ts)
                    extracted_count += 1
                    
                frame_idx += 1
                if self.cfg.max_frames and extracted_count >= self.cfg.max_frames:
                    break
        finally:
            if isinstance(source, (str, int)):
                cap.release()
                
        return timestamps

    def _open_capture(self, source: Union[str, int, cv2.VideoCapture]) -> cv2.VideoCapture:
        if isinstance(source, cv2.VideoCapture):
            # Reset position if it's an already open capture
            source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source!r}")
        return cap
