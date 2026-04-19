import pytest
import numpy as np
from kpe.extract.optical_flow import FlowConfig

def test_flow_config_defaults():
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
