from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dtaidistance import dtw
import tempfile
import os
import numpy as np
import sys

# Make kpe package importable regardless of working directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kpe.extract.optical_flow import SparseOpticalFlowExtractor, FlowConfig

app = FastAPI()

# ---------------------------------------------------------------------------
# Reference DB (placeholder — replace with real signatures from karthik's pipeline)
# ---------------------------------------------------------------------------

# Shape of each reference: (T, descriptor_dim) — but stored flat for DTW 1-D distance.
# Until real reference signatures are loaded, we keep the old mock so /match still works.
_cfg = FlowConfig()
DESCRIPTOR_DIM = _cfg.descriptor_dim  # == 10 by default, driven by n_magnitude_bins + n_angle_bins + 2

demoval = [
    [0.1 * i for i in range(20)],
    [0.9 * i for i in range(20)],
]

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
    Write video bytes to a temp file, run SparseOpticalFlowExtractor,
    build the S(t) motion signature (T, descriptor_dim) via build_signature(),
    then flatten to a 1-D Python list of float32 values for JSON + DTW.

    The descriptor_dim is read from FlowConfig.descriptor_dim — not hardcoded —
    so it stays correct if bin counts change upstream.
    """
    extractor = SparseOpticalFlowExtractor(config=FlowConfig())

    # cv2.VideoCapture needs a real file path, not an in-memory buffer
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        descriptors = extractor.process_video(tmp_path)

        if not descriptors:
            # No frames could be processed — return zeros so downstream doesn't crash
            return [0.0] * DESCRIPTOR_DIM

        # build_signature returns shape (T, descriptor_dim) as float32
        # fill_invalid=True keeps temporal alignment intact (DTW-safe zero vectors for bad frames)
        signature_2d: np.ndarray = extractor.build_signature(
            descriptors,
            fill_invalid=True,
        )

        # Flatten (T, 10) → (T*10,) then cast to plain Python floats for JSON serialisation.
        # The sliding_dtw call will operate on this 1-D representation.
        # NOTE: if frame_skip is later tuned for clip length, signature_2d.shape[0] (T)
        # will change — that's fine, sliding_dtw already handles variable-length queries.
        return signature_2d.flatten().astype(float).tolist()

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# DTW matching
# ---------------------------------------------------------------------------

def sliding_dtw(query: list[float], ref: list[float]) -> float:
    """
    Slide a window equal to the query length across the reference and return
    the minimum DTW distance found. Handles query longer than ref gracefully.
    """
    window_size = len(query)

    if len(ref) < window_size:
        return dtw.distance(query, ref)

    best = float("inf")
    for i in range(len(ref) - window_size + 1):
        window = ref[i : i + window_size]
        score = dtw.distance(query, window)
        if score < best:
            best = score

    return best


def find_best_match(query_signature: list[float]) -> tuple[int, float]:
    best_score = float("inf")
    best_id = -1

    for i, ref in enumerate(demoval):
        score = sliding_dtw(query_signature, ref)
        if score < best_score:
            best_score = score
            best_id = i

    # TODO: calibrate this formula once real reference signatures are in demoval.
    # Current 1/(1+score) is a placeholder — score magnitude will change significantly
    # with real (T*10)-length vectors vs the old 20-element mocks.
    confidence = 1 / (1 + best_score)
    return best_id, confidence


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Sab Khairiyat"}


@app.post("/match")
def match_vid(inputData: InputVid):
    """Manual signature submission — useful for testing without a video file."""
    best_id, confidence = find_best_match(inputData.query_signature)
    return {
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH",
    }


@app.post("/verify")
async def verify_video(file: UploadFile = File(...)):
    """
    Accept a video upload, extract its motion signature via KPE optical flow,
    and return the DTW match result plus both signals for the dashboard waveform.
    """
    video_bytes = await file.read()

    if not video_bytes:
        return {"error": "Empty file"}

    # Raised from 5 MB — most real MP4 clips will exceed that.
    # 200 MB covers typical broadcast clips; tune further if needed.
    MAX_BYTES = 200 * 1024 * 1024  # 200 MB
    if len(video_bytes) > MAX_BYTES:
        return {"error": f"File too large (limit {MAX_BYTES // (1024*1024)} MB)"}

    query_signature = extract_signature(video_bytes)
    best_id, confidence = find_best_match(query_signature)

    return {
        "file_name": file.filename,
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH",
        "query_signal": query_signature,
        "ref_signal": demoval[best_id],
    }