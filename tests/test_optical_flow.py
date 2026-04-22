import time
import pytest
import numpy as np
import cv2
import os
from kpe.extract.optical_flow import FlowConfig, SparseOpticalFlowExtractor

def test_flow_config_defaults() -> None:
    config = FlowConfig()
    assert config.descriptor_dim == 10
    assert isinstance(config.frame_width, int)
    assert isinstance(config.frame_height, int)
    assert isinstance(config.max_corners, int)
    assert isinstance(config.quality_level, float)
    assert isinstance(config.min_distance, float)
    assert isinstance(config.block_size, int)
    assert isinstance(config.lk_win_size, tuple)
    assert isinstance(config.lk_max_level, int)
    assert isinstance(config.lk_max_iter, int)
    assert isinstance(config.lk_epsilon, float)
    assert isinstance(config.n_magnitude_bins, int)
    assert isinstance(config.n_angle_bins, int)
    assert isinstance(config.magnitude_clip, float)
    assert isinstance(config.min_tracked_points, int)

def test_detect_corners_synthetic() -> None:
    y, x = np.indices((240, 320))
    frame = (((x // 20) + (y // 20)) % 2 * 255).astype(np.uint8)
    config = FlowConfig(max_corners=200)
    extractor = SparseOpticalFlowExtractor(config)
    descriptors = extractor.process_frames([frame, frame])
    assert len(descriptors) == 1
    desc = descriptors[0]
    assert desc.vector.shape == (10,)
    assert desc.n_points >= 0
    assert desc.valid

def test_normal_tracking_horizontal_pan():
    """Validates descriptor outputs a strong mean_dx when moving frames horizontally."""
    y, x = np.indices((240, 320))
    frame1 = (((x // 20) + (y // 20)) % 2 * 255).astype(np.uint8)
    
    # Shift image left by 5 pixels (simulating camera pan right)
    matrix = np.float32([[1, 0, -5], [0, 1, 0]])
    frame2 = cv2.warpAffine(frame1, matrix, (320, 240))
    
    config = FlowConfig(max_corners=200)
    extractor = SparseOpticalFlowExtractor(config)
    descriptors = extractor.process_frames([frame1, frame2])
    
    desc = descriptors[0]
    assert desc.valid
    # The optical flow vectors should represent motion of points: p1 - p0
    # Since we shifted image left, old point at x moved to x-5.
    # Therefore dx should be approx -5.
    mean_dx = desc.vector[8]
    mean_dy = desc.vector[9]
    assert mean_dx < -3.0  # Approx -5, maybe less due to subpixel
    assert abs(mean_dy) < 1.0

from unittest.mock import patch

def test_scene_cut_redetection_fallback():
    """Tests if tracking breaks completely across a scene cut, triggering invalidation and redetection fallback."""
    y, x = np.indices((240, 320))
    frame1 = (((x // 20) + (y // 20)) % 2 * 255).astype(np.uint8)
    frame2 = frame1.copy()
    
    config = FlowConfig(min_tracked_points=15)
    extractor = SparseOpticalFlowExtractor(config)
    
    # Mock cv2.calcOpticalFlowPyrLK to simulate complete tracking failure (status all 0s)
    with patch('cv2.calcOpticalFlowPyrLK') as mock_lk:
        # Return signature: nextPts, status, err
        # Simulate returning 200 points but all have status 0
        mock_lk.return_value = (
            np.zeros((200, 1, 2), dtype=np.float32), 
            np.zeros((200, 1), dtype=np.uint8), 
            np.zeros((200, 1), dtype=np.float32)
        )
        descriptors = extractor.process_frames([frame1, frame2])
    
    assert len(descriptors) == 1
    desc = descriptors[0]
    assert not desc.valid
    assert desc.n_points == 0

def test_low_texture_frame():
    """Tests behaviour when Shi-Tomasi finds no corners."""
    frame1 = np.ones((240, 320), dtype=np.uint8) * 128
    frame2 = np.ones((240, 320), dtype=np.uint8) * 128
    
    config = FlowConfig()
    extractor = SparseOpticalFlowExtractor(config)
    descriptors = extractor.process_frames([frame1, frame2])
    
    assert len(descriptors) == 1
    desc = descriptors[0]
    assert not desc.valid

def test_process_video_integration():
    """Verify process_video with dummy mp4 if exists."""
    video_path = os.path.join(os.path.dirname(__file__), '..', 'dummy.mp4')
    if not os.path.exists(video_path):
        pytest.skip(f"Test video not found: {video_path}")
        
    config = FlowConfig(min_tracked_points=10)
    extractor = SparseOpticalFlowExtractor(config)
    
    t0 = time.perf_counter()
    descriptors = extractor.process_video(video_path)
    t1 = time.perf_counter()
    
    assert len(descriptors) > 0
    valid_count = sum(d.valid for d in descriptors)
    
    # Verify performance: check if average frame time <= 33ms (30fps budget)
    frames_processed = len(descriptors) + 1
    ms_per_frame = (t1 - t0) * 1000 / frames_processed
    assert ms_per_frame < 35.0, f"Performance budget exceeded: {ms_per_frame:.2f} ms/frame"

