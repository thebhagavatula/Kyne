from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dtaidistance import dtw
import tempfile
import os
import numpy as np
import sys
import json
from pathlib import Path

# Make kpe package importable regardless of working directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kpe.extract.optical_flow import SparseOpticalFlowExtractor, FlowConfig

app = FastAPI()

# ---------------------------------------------------------------------------
# Reference DB (real signatures from your local generated reference_signatures.json)
# ---------------------------------------------------------------------------

_cfg = FlowConfig()
DESCRIPTOR_DIM = _cfg.descriptor_dim

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "reference_signatures.json"

if DATA_FILE.exists():
    with open(DATA_FILE, "r") as f:
        demoval = json.load(f)
    print(f"Loaded {len(demoval)} reference signatures")
else:
    demoval = []
    print("Warning: reference_signatures.json not found. demoval is empty.")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class InputVid(BaseModel):
    query_signature: List[float]
    video_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def extract_signature(video_bytes: bytes) -> list[float]:
    """
    Convert uploaded video bytes into motion signature.
    """

    print("Starting signature extraction...")

    extractor = SparseOpticalFlowExtractor(config=FlowConfig())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        descriptors = extractor.process_video(tmp_path)

        print("Optical flow extraction complete")

        if not descriptors:
            print("No descriptors found")
            return [0.0] * DESCRIPTOR_DIM

        signature_2d: np.ndarray = extractor.build_signature(
            descriptors,
            fill_invalid=True
        )

        print("Signature build complete")

        return signature_2d.flatten().astype(float).tolist()

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# DTW matching
# ---------------------------------------------------------------------------

def sliding_dtw(query: list[float], ref: list[float]) -> float:
    """
    Compare query signature against one reference signature.
    """

    window_size = len(query)

    if len(ref) < window_size:
        return dtw.distance(query, ref)

    best = float("inf")

    for i in range(len(ref) - window_size + 1):
        window = ref[i:i + window_size]
        score = dtw.distance(query, window)

        if score < best:
            best = score

    return best


def find_best_match(query_signature: list[float]) -> tuple[int, float]:
    """
    Compare query against all reference signatures and return best match.
    """

    if not demoval:
        return -1, 0.0

    best_score = float("inf")
    best_id = -1

    print("Starting DTW matching...")

    for i, ref in enumerate(demoval):
        score = sliding_dtw(query_signature, ref)

        print(f"Compared with reference {i}: score={score}")

        if score < best_score:
            best_score = score
            best_id = i

    confidence = 1 / (1 + best_score)

    print(f"Best match: {best_id}, confidence: {confidence}")

    return best_id, confidence


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Sab Khairiyat"}


@app.post("/match")
def match_vid(inputData: InputVid):
    """
    Manual signature submission (JSON-based testing).
    """

    best_id, confidence = find_best_match(inputData.query_signature)

    return {
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH",
    }


@app.post("/verify")
async def verify_video(file: UploadFile = File(...)):
    """
    Upload video → extract signature → compare against reference DB.
    """

    print("Received file upload")

    video_bytes = await file.read()

    print("File read complete")

    if not video_bytes:
        return {"error": "Empty file"}

    MAX_BYTES = 200 * 1024 * 1024

    if len(video_bytes) > MAX_BYTES:
        return {
            "error": f"File too large (limit {MAX_BYTES // (1024 * 1024)} MB)"
        }

    query_signature = extract_signature(video_bytes)

    print("Signature extraction complete")

    best_id, confidence = find_best_match(query_signature)

    print("Matching complete")

    if best_id == -1:
        return {
            "error": "No reference database loaded"
        }

    return {
        "file_name": file.filename,
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH",
        "query_signal": query_signature,
        "ref_signal": demoval[best_id],
    }