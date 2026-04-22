"""
tests/test_pipeline_day5.py

Integration tests for Deliverable 3 (Day 5 pipeline validation).
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from kpe.extract.preprocessor import PreprocessorConfig, VideoPreprocessor
from kpe.extract.optical_flow import FlowConfig, SparseOpticalFlowExtractor
from kpe.store.signature_store import StoreConfig, SignatureStore


def test_pipeline_integration():
    """
    Tests the end-to-end extraction and storage pipeline using synthetic in-memory frames.
    """
    # 1. Generate synthetic video clip (moving rectangle on black background)
    frames = []
    width, height = 640, 480
    for i in range(20):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Moving white rectangle creates texture for Shi-Tomasi/LK tracking
        start_x, start_y = 50 + i * 5, 50 + i * 5
        cv2.rectangle(frame, (start_x, start_y), (start_x + 100, start_y + 100), (255, 255, 255), -1)
        frames.append(frame)
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 2. Preprocess
        # Skips every other frame, targets standard 320x240
        prep_config = PreprocessorConfig(target_width=320, target_height=240, frame_skip=2)
        preprocessor = VideoPreprocessor(prep_config)
        
        prep_frames = preprocessor.preprocess(frames)
        timestamps = preprocessor.frame_timestamps(frames)
        
        assert len(prep_frames) == 10  # 20 frames / 2 = 10 frames
        assert len(timestamps) == 10
        assert prep_frames[0].shape == (240, 320)  # grayscale and resized correctly
        
        # 3. Process frames (Optical Flow)
        flow_config = FlowConfig()
        extractor = SparseOpticalFlowExtractor(flow_config)
        descriptors = extractor.process_frames(prep_frames)
        
        # 4. Build Signature
        signature = extractor.build_signature(descriptors)
        
        # We fed 10 frames into the extractor, so it generates 9 transition descriptors
        T = signature.shape[0]
        assert T == 9
        assert signature.shape == (T, 10)
        
        # 5. Store signature
        store_config = StoreConfig(store_dir=str(tmpdir_path / "signatures"))
        store = SignatureStore(store_config)
        
        clip_id = "test_clip_123"
        metadata = {"test_run": True, "source": "synthetic"}
        
        npy_path = store.save(clip_id, signature, metadata)
        assert npy_path.exists()
        
        # 6. Load signature
        loaded_sig, loaded_manifest = store.load(clip_id)
        assert np.array_equal(signature, loaded_sig)
        assert loaded_manifest["clip_id"] == clip_id
        assert loaded_manifest["frame_count"] == T
        assert loaded_manifest["test_run"] is True
        
        assert clip_id in store.list_clips()
        
        # 7. to_json_payload
        payload = store.to_json_payload(clip_id, signature, metadata)
        
        # assert JSON serializable natively
        payload_json = json.dumps(payload)
        assert isinstance(payload_json, str)
        assert payload["clip_id"] == clip_id
        assert payload["shape"] == [T, 10]
        assert isinstance(payload["signature"], list)
