import argparse
import sys
import os
from pathlib import Path

# Ensure kpe module is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from kpe.extract.optical_flow import FlowConfig, SparseOpticalFlowExtractor

def main() -> None:
    """
    Main entry point for the optical flow smoke test script.
    
    Parses CLI arguments, instantiates the FlowConfig and SparseOpticalFlowExtractor,
    processes the input video, calculates performance bounds, and saves the S(t) signature.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Smoke test for optical flow extractor.")

    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--max-corners", type=int, default=None, help="Max feature points to track")
    parser.add_argument("--quality-level", type=float, default=None, help="Min accepted quality ratio (0-1)")
    parser.add_argument("--lk-max-level", type=int, default=None, help="Pyramid levels for LK")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
        
    config_kwargs = {}
    if args.max_corners is not None:
        config_kwargs["max_corners"] = args.max_corners
    if args.quality_level is not None:
        config_kwargs["quality_level"] = args.quality_level
    if args.lk_max_level is not None:
        config_kwargs["lk_max_level"] = args.lk_max_level
        
    try:
        config = FlowConfig(**config_kwargs)
        extractor = SparseOpticalFlowExtractor(config)
        
        descriptors = extractor.process_video(str(video_path))
        
        total_frames = len(descriptors)
        valid_descriptors = [d for d in descriptors if d.valid]
        valid_count = len(valid_descriptors)
        invalid_count = total_frames - valid_count
        
        mean_points = 0.0
        if valid_count > 0:
            mean_points = sum(d.n_points for d in valid_descriptors) / valid_count
            
        print(f"Total frames processed: {total_frames}")
        print(f"Valid descriptors: {valid_count}")
        print(f"Invalid descriptors: {invalid_count}")
        print(f"Mean n_points per valid frame: {mean_points:.2f}")
        
        if total_frames > 0:
            signature = extractor.build_signature(descriptors)
            out_path = video_path.with_suffix(".npy")
            np.save(str(out_path), signature)
            print(f"Saved signature to {out_path}")
        else:
            print("No frames were processed. Signature not saved.")
            
    except cv2.error as e:
        print(f"OpenCV Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"IO Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main()
