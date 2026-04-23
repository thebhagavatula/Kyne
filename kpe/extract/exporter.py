"""
kpe/extract/exporter.py

Handles exporting and importing of motion signatures.
"""

import json
import numpy as np


class SignatureExporter:
    """
    Serialises and deserialises KPE motion signatures.
    """

    @staticmethod
    def to_json(sig: np.ndarray, metadata: dict) -> str:
        """
        Export signature and metadata to a JSON string.
        """
        payload = {
            "signature": sig.tolist(),
            "metadata": metadata,
        }
        return json.dumps(payload)

    @staticmethod
    def to_vector_stream(sig: np.ndarray) -> list[list[float]]:
        """
        Convert signature array to a nested Python list.
        """
        return sig.tolist()

    @staticmethod
    def from_json(payload: str) -> tuple[np.ndarray, dict]:
        """
        Load signature and metadata from a JSON string.
        """
        data = json.loads(payload)
        sig = np.array(data["signature"], dtype=np.float32)
        return sig, data.get("metadata", {})
