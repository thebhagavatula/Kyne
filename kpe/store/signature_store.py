"""
kpe/store/signature_store.py

SignatureStore handles serialisation and deserialisation of S(t) signatures.
Layer: Store (Layer 3 of the KPE pipeline)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StoreConfig:
    """Configuration for signature storage."""
    store_dir: str = "./signatures/"

class SignatureStore:
    """
    Handles storage and retrieval of motion signatures and metadata.
    """
    def __init__(self, config: StoreConfig = None):
        self.cfg = config or StoreConfig()
        self.store_path = Path(self.cfg.store_dir)
        self.store_path.mkdir(parents=True, exist_ok=True)
        logger.info("SignatureStore initialised at %s", self.store_path.absolute())

    def save(self, clip_id: str, signature: np.ndarray, metadata: dict) -> Path:
        """
        Saves the signature array and a sidecar JSON manifest.
        
        Returns:
            Path: The path to the saved .npy file.
        """
        npy_file = self.store_path / f"{clip_id}.npy"
        json_file = self.store_path / f"{clip_id}.json"
        
        np.save(str(npy_file), signature)
        
        manifest = {
            "clip_id": clip_id,
            "shape": list(signature.shape),
            "descriptor_dim": signature.shape[1] if signature.ndim > 1 else 0,
            "frame_count": signature.shape[0] if signature.ndim > 0 else 0,
            "timestamp_saved": datetime.now(timezone.utc).isoformat(),
        }
        manifest.update(metadata)
        
        with open(json_file, "w") as f:
            json.dump(manifest, f, indent=4)
            
        logger.info("Saved signature %s and manifest to %s", clip_id, self.store_path)
        return npy_file

    def load(self, clip_id: str) -> Tuple[np.ndarray, dict]:
        """
        Loads the signature array and its manifest.
        """
        npy_file = self.store_path / f"{clip_id}.npy"
        json_file = self.store_path / f"{clip_id}.json"
        
        if not npy_file.exists() or not json_file.exists():
            raise FileNotFoundError(f"Signature or manifest for {clip_id} not found in {self.store_path}")
            
        signature = np.load(str(npy_file))
        
        with open(json_file, "r") as f:
            manifest = json.load(f)
            
        return signature, manifest

    def list_clips(self) -> List[str]:
        """
        Returns all stored clip IDs based on available .npy files.
        """
        clips = []
        for file_path in self.store_path.glob("*.npy"):
            clips.append(file_path.stem)
        return sorted(clips)

    def to_json_payload(self, clip_id: str, signature: np.ndarray, metadata: dict) -> dict:
        """
        Returns a JSON-serialisable dict representation of the signature as a nested list 
        for API transport (for the FastAPI /verify endpoint).
        """
        payload = {
            "clip_id": clip_id,
            "shape": list(signature.shape),
            "descriptor_dim": signature.shape[1] if signature.ndim > 1 else 0,
            "frame_count": signature.shape[0] if signature.ndim > 0 else 0,
            "signature": signature.tolist()
        }
        payload.update(metadata)
        return payload
