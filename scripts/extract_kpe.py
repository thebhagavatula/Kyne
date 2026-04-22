"""
scripts/extract_kpe.py

CLI extraction script for the KPE Layer 2 (Extract) pipeline.
Ingests a video, calculates the sparse optical flow motion signature S(t),
and exports the result (.npy) and metadata manifest (.json).
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import cv2
import numpy as np

# Ensure kpe module can be found
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))

from kpe.extract.optical_flow import FlowConfig, SparseOpticalFlowExtractor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="KPE Layer 2 - Motion Signature Extraction")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npy and .json outputs")
    parser.add_argument("--debug", action="store_true", help="Enable tracking visualisation")
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_hash = input_path.stem  # Using filename stem as hash for simplicity
    npy_path = output_dir / f"{video_hash}.npy"
    json_path = output_dir / f"{video_hash}.json"
    
    logger.info("Extracting motion signature for: %s", input_path.name)
    
    # KPE Layer 2 Extraction
    t0 = time.perf_counter()
    
    config = FlowConfig()
    extractor = SparseOpticalFlowExtractor(config, debug=args.debug)
    descriptors = extractor.process_video(str(input_path))
    
    # 30fps budget check proxy
    t1 = time.perf_counter()
    n_frames = len(descriptors) + 1
    duration_sec = t1 - t0
    
    if n_frames < 2:
        logger.error("Video too short or unreadable.")
        sys.exit(1)
        
    logger.info("Tracking finished in %.2fs (%.2f ms/frame)", duration_sec, (duration_sec/n_frames)*1000)
    
    # Build Signature (Day 3 upgrades built-in)
    sig = extractor.build_signature(descriptors)
    
    # Valid metrics calculation
    valid_count = sum(d.valid for d in descriptors)
    invalid_ratio = 1.0 - (valid_count / len(descriptors))
    
    # Storage
    np.save(str(npy_path), sig)
    logger.info("Saved signature -> %s", npy_path)
    
    # Manifest creation for Layer 3 (Store)
    manifest = {
        "video_hash": video_hash,
        "input_filename": input_path.name,
        "frames_processed": n_frames,
        "signature_shape": list(sig.shape),
        "invalid_ratio": round(invalid_ratio, 4),
        "descriptor_dim": config.descriptor_dim,
        "extraction_time_ms_per_frame": round((duration_sec / n_frames) * 1000, 2)
    }
    
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    logger.info("Saved manifest -> %s", json_path)


if __name__ == "__main__":
    main()
