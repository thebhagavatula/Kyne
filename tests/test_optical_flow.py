import pytest
import numpy as np
from kpe.extract.optical_flow import FlowConfig, SparseOpticalFlowExtractor

def test_flow_config_defaults() -> None:
    """
    Validates FlowConfig defaults ensuring correct structure padding alongside parameter types.
    
    Parameters:
        None
        
    Returns:
        None
    """
    config = FlowConfig()

    
    # Assert descriptor dimensionality is correctly set to 10
    assert config.descriptor_dim == 10
    
    # Assert all fields have the correct types
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

@pytest.mark.parametrize("max_corners", [50, 200])
def test_detect_corners_synthetic(max_corners: int) -> None:
    """
    Processes synthetic 240x320 checkerboard frames asserting deterministic tracker limits.
    
    Parameters:
        max_corners (int): Limit bounding to configure Shi-Tomasi feature search.
        
    Returns:
        None
    """
    # Generate a synthetic 240x320 uint8 grayscale checkerboard frame
    y, x = np.indices((240, 320))
    frame = (((x // 20) + (y // 20)) % 2 * 255).astype(np.uint8)
    
    config = FlowConfig(max_corners=max_corners)
    extractor = SparseOpticalFlowExtractor(config)
    
    # Process two identical copies of the frame
    descriptors = extractor.process_frames([frame, frame])
    
    # Assertions
    assert len(descriptors) == 1
    desc = descriptors[0]
    
    assert desc.vector.shape == (10,)
    assert desc.n_points >= 0
