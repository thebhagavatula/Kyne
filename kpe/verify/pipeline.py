"""
kpe/verify/pipeline.py

Integration pipeline connecting extraction to downstream API and DTW tasks.
"""

from typing import Callable, Optional
import numpy as np

from kpe.extract.optical_flow import SparseOpticalFlowExtractor, FlowConfig
from kpe.extract.exporter import SignatureExporter


def run_extraction(
    source,
    config: FlowConfig,
    matcher_fn: Optional[Callable[[np.ndarray, np.ndarray], dict]] = None
) -> dict:
    """
    Runs the full KPE extraction pipeline on a video source.
    Accepts an optional matcher_fn to allow downstream DTW integration without
    modifying this file.
    """
    extractor = SparseOpticalFlowExtractor(config)
    descriptors = extractor.process_video(source)

    if not descriptors:
        return {
            "signature_json": None,
            "metadata": {"error": "No frames processed"},
            "frame_count": 0,
            "valid_frames": 0,
        }

    signature = extractor.build_signature(descriptors)

    frame_count = len(descriptors)
    valid_frames = sum(1 for d in descriptors if d.valid)

    metadata = {
        "frame_count": frame_count,
        "valid_frames": valid_frames,
        "descriptor_dim": config.descriptor_dim,
    }

    signature_json = SignatureExporter.to_json(signature, metadata)

    return {
        "signature_json": signature_json,
        "metadata": metadata,
        "frame_count": frame_count,
        "valid_frames": valid_frames,
    }
