"""
tests/test_pipeline.py

Integration tests for the KPE extraction pipeline using synthetic arrays.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kpe.extract.optical_flow import FlowConfig
from kpe.verify.pipeline import run_extraction


def _generate_synthetic_frames(num_frames: int, width: int = 320, height: int = 240) -> list[np.ndarray]:
    """Generate noisy synthetic grayscale frames for testing."""
    frames = []
    for _ in range(num_frames):
        # Add high variance noise so Shi-Tomasi finds corners
        frame = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        frames.append(frame)
    return frames


class MockVideoCapture:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0

    def read(self):
        if self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame
        return False, None

    def grab(self):
        if self.idx < len(self.frames):
            self.idx += 1
            return True
        return False

    def isOpened(self):
        return True

    def release(self):
        pass


@patch("kpe.extract.optical_flow.SparseOpticalFlowExtractor._open_capture")
def test_run_extraction_happy_path(mock_open_capture):
    """Test standard extraction pipeline with a mocked matcher_fn."""
    frames = _generate_synthetic_frames(10)
    mock_open_capture.return_value = MockVideoCapture(frames)

    config = FlowConfig(max_corners=50)

    # Mock matcher_fn as scaffolding for future DTW integration
    matcher_fn = MagicMock(return_value={"distance": 0.5, "verdict": "match"})

    result = run_extraction("dummy.mp4", config, matcher_fn=matcher_fn)

    assert result["frame_count"] == 9
    assert "signature_json" in result
    assert result["signature_json"] is not None

    data = json.loads(result["signature_json"])
    assert "signature" in data
    assert len(data["signature"]) == 9
    assert "metadata" in data


@patch("kpe.extract.optical_flow.SparseOpticalFlowExtractor._open_capture")
def test_run_extraction_frame_skip(mock_open_capture):
    """Test extraction pipeline correctly applies frame skipping."""
    frames = _generate_synthetic_frames(10)
    mock_open_capture.return_value = MockVideoCapture(frames)

    config = FlowConfig(frame_skip=2, max_corners=50)

    result = run_extraction("dummy.mp4", config)

    # 10 frames total. skip=2 means we process every 2nd frame.
    # pairs: (0, 2), (2, 4), (4, 6), (6, 8) -> 4 descriptors
    assert result["frame_count"] == 4


@patch("kpe.extract.optical_flow.SparseOpticalFlowExtractor._open_capture")
def test_run_extraction_empty_source(mock_open_capture):
    """Test guard logic when the source contains no frames."""
    mock_open_capture.return_value = MockVideoCapture([])

    config = FlowConfig()
    result = run_extraction("empty.mp4", config)

    assert result["frame_count"] == 0
    assert result["signature_json"] is None
    assert result["metadata"]["error"] == "No frames processed"


@patch("kpe.extract.optical_flow.SparseOpticalFlowExtractor._open_capture")
@patch.dict("os.environ", {"USE_GEMINI": "true"}, clear=True)
@patch("kpe.ai.gemini_client.GeminiClient.analyze_signature")
def test_run_extraction_with_gemini(mock_analyze, mock_open_capture):
    """Test extraction pipeline correctly invokes Gemini when enabled."""
    frames = _generate_synthetic_frames(5)
    mock_open_capture.return_value = MockVideoCapture(frames)
    mock_analyze.return_value = "This is a mocked Gemini insight."

    config = FlowConfig(max_corners=50)
    result = run_extraction("dummy.mp4", config)

    assert result["gemini_insights"] == "This is a mocked Gemini insight."
    mock_analyze.assert_called_once()


@patch("kpe.extract.optical_flow.SparseOpticalFlowExtractor._open_capture")
@patch.dict("os.environ", {"USE_GEMINI": "false"}, clear=True)
@patch("kpe.ai.gemini_client.GeminiClient.analyze_signature")
def test_run_extraction_without_gemini(mock_analyze, mock_open_capture):
    """Test extraction pipeline does not invoke Gemini when disabled."""
    frames = _generate_synthetic_frames(5)
    mock_open_capture.return_value = MockVideoCapture(frames)

    config = FlowConfig(max_corners=50)
    result = run_extraction("dummy.mp4", config)

    assert result["gemini_insights"] is None
    mock_analyze.assert_not_called()
