from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dtaidistance import dtw_ndim
import tempfile
import os
import numpy as np
import sys
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kpe.extract.optical_flow import SparseOpticalFlowExtractor, FlowConfig
from kpe.ai.gemini_client import GeminiClient

app = FastAPI()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg = FlowConfig()
DESCRIPTOR_DIM = _cfg.descriptor_dim  # 10

TARGET_FRAMES = 120        # raised from 60 — more detail, still fast enough
SAKOE_CHIBA_FRAC = 0.2    # loosened from 0.1 — allows more warp on re-encoded clips
ANALOG_HOLE_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Reference DB
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "reference_signatures.json"

MATCH_THRESHOLD = 4.5


def _downsample(sig: np.ndarray, target: int) -> np.ndarray:
    """
    Temporally downsample (T, D) → (target, D) via linear interpolation.
    """
    T = len(sig)
    if T <= target:
        return sig
    indices = np.linspace(0, T - 1, target)
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (indices - lo)[:, None]
    return ((1 - frac) * sig[lo] + frac * sig[hi]).astype(np.float32)


def _load_references(path: Path) -> list[np.ndarray]:
    """
    Load reference signatures as (T, 10) float32 arrays.
    Downsamples at load time — paid once, not per request.
    L2 normalisation is OFF — raw descriptor values give DTW more to work with.
    """
    if not path.exists():
        print("Warning: reference_signatures.json not found. demoval is empty.")
        return []
    
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        print("Warning: reference_signatures.json is empty or corrupt. demoval is empty.")
        return []

    refs = []
    for entry in raw:
        arr = np.array(entry, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, DESCRIPTOR_DIM)
        arr = _downsample(arr, TARGET_FRAMES)
        refs.append(arr)

    print(f"Loaded {len(refs)} reference signatures — shape {refs[0].shape if refs else 'n/a'}")
    return refs


demoval: list[np.ndarray] = _load_references(DATA_FILE)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _compute_analog_hole_confidence(
    descriptors: list,
    signature: np.ndarray,
    max_corners: int,
) -> tuple[float, dict]:
    """
    Heuristic confidence that a clip came through an analog hole
    (e.g. phone camera recording another screen).
    """
    if not descriptors:
        return 0.0, {
            "invalid_ratio": 1.0,
            "track_density": 0.0,
            "motion_jitter": 0.0,
            "signal_flicker": 0.0,
        }

    n_desc = len(descriptors)
    invalid_ratio = sum(not d.valid for d in descriptors) / n_desc
    mean_tracked = float(np.mean([d.n_points for d in descriptors]))
    track_density = _clip01(mean_tracked / max(1, max_corners))

    if len(signature) > 1:
        global_motion = signature[:, -2:]
        motion_deltas = np.diff(global_motion, axis=0)
        motion_jitter = float(np.median(np.linalg.norm(motion_deltas, axis=1)))

        signal_energy = np.linalg.norm(signature, axis=1)
        signal_flicker = float(np.median(np.abs(np.diff(signal_energy))))
    else:
        motion_jitter = 0.0
        signal_flicker = 0.0

    invalid_score = _clip01(invalid_ratio / 0.40)
    low_track_score = _clip01((0.40 - track_density) / 0.40)
    jitter_score = _clip01(motion_jitter / 1.50)
    flicker_score = _clip01(signal_flicker / 0.25)

    confidence = (
        0.35 * invalid_score
        + 0.25 * low_track_score
        + 0.25 * jitter_score
        + 0.15 * flicker_score
    )

    metrics = {
        "invalid_ratio": float(invalid_ratio),
        "track_density": float(track_density),
        "motion_jitter": float(motion_jitter),
        "signal_flicker": float(signal_flicker),
    }
    return _clip01(confidence), metrics


def _compute_analog_hole_confidence_from_signature(signature: np.ndarray) -> float:
    if len(signature) <= 1:
        return 0.0
    global_motion = signature[:, -2:]
    motion_deltas = np.diff(global_motion, axis=0)
    motion_jitter = float(np.median(np.linalg.norm(motion_deltas, axis=1)))
    signal_energy = np.linalg.norm(signature, axis=1)
    signal_flicker = float(np.median(np.abs(np.diff(signal_energy))))
    return _clip01(0.65 * _clip01(motion_jitter / 1.50) + 0.35 * _clip01(signal_flicker / 0.25))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class InputVid(BaseModel):
    query_signature: List[float]
    video_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def extract_signature(video_bytes: bytes) -> tuple[np.ndarray, float, dict]:
    """
    Extract (T, 10) float32 motion signature from raw video bytes.
    Returns 2D ndarray plus analog-hole confidence and contributing metrics.
    Raw values give DTW the variance it needs to discriminate between videos.
    """
    print("Starting signature extraction...")
    extractor = SparseOpticalFlowExtractor(config=FlowConfig())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        try:
            descriptors = extractor.process_video(tmp_path)
        except OSError as e:
            print(f"Failed to process video: {e}")
            return np.zeros((1, DESCRIPTOR_DIM), dtype=np.float32), 0.0, {}

        if not descriptors:
            print("No descriptors found — returning zeros")
            return np.zeros((1, DESCRIPTOR_DIM), dtype=np.float32), 0.0, {}

        sig: np.ndarray = extractor.build_signature(
            descriptors,
            fill_invalid=True,
            l2_normalize=False,   # OFF — L2 norm collapses inter-video variance
        )
        sig = _downsample(sig, TARGET_FRAMES)
        analog_confidence, analog_metrics = _compute_analog_hole_confidence(
            descriptors=descriptors,
            signature=sig,
            max_corners=_cfg.max_corners,
        )
        print(f"Signature shape after downsample: {sig.shape}")
        return sig, analog_confidence, analog_metrics

    finally:
        os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# DTW matching
# ---------------------------------------------------------------------------

def sliding_dtw_nd(query: np.ndarray, ref: np.ndarray) -> tuple[float, int]:
    """
    Multivariate sliding DTW with Sakoe-Chiba band.
    query : (T_q, 10), ref : (T_r, 10)
    """
    T_q = len(query)
    T_r = len(ref)
    band = max(1, int(T_q * SAKOE_CHIBA_FRAC))

    if T_r < T_q:
        return dtw_ndim.distance(query, ref, window=band), 0

    best = float("inf")
    best_idx = 0
    for i in range(T_r - T_q + 1):
        score = dtw_ndim.distance(query, ref[i : i + T_q], window=band)
        if score < best:
            best = score
            best_idx = i

    return best, best_idx


def find_best_match(query_sig: np.ndarray) -> tuple[int, float, float, str, list, list]:
    if not demoval:
        return -1, 0.0, float("inf"), "NO MATCH", [], []

    scores = []
    start_indices = []
    print("Starting DTW matching...")

    for i, ref in enumerate(demoval):
        score, start_idx = sliding_dtw_nd(query_sig, ref)
        scores.append(score)
        start_indices.append(start_idx)
        print(f"  ref {i}: score={score:.4f}, start={start_idx}")

    scores = np.array(scores)
    best_id = int(np.argmin(scores))
    best_score = float(scores[best_id])
    best_start = start_indices[best_id]

    # Map the DTW distance to a 0-1 confidence score using the empirical MATCH_THRESHOLD
    confidence = float(np.clip(1.0 - (best_score / MATCH_THRESHOLD), 0.0, 1.0))

    # verdict driven by raw score, not confidence
    verdict = "MATCH" if best_score < MATCH_THRESHOLD else "NO MATCH"
    
    # Calculate Warping Path for the best match
    T_q = len(query_sig)
    band = max(1, int(T_q * SAKOE_CHIBA_FRAC))
    best_ref = demoval[best_id]
    
    if len(best_ref) < T_q:
        path = dtw_ndim.warping_path(query_sig, best_ref, window=band)
    else:
        path = dtw_ndim.warping_path(query_sig, best_ref[best_start : best_start + T_q], window=band)
        
    dtw_path_y = [p[0] for p in path] # query (suspect) frame
    dtw_path_x = [(p[1] + best_start) for p in path] # ref frame (offset by window start)

    print(f"Best match: ref {best_id}, raw_score={best_score:.4f}, verdict={verdict}")
    return best_id, confidence, best_score, verdict, dtw_path_x, dtw_path_y

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Sab Khairiyat"}


@app.post("/match")
def match_vid(inputData: InputVid):
    arr = np.array(inputData.query_signature, dtype=np.float32).reshape(-1, DESCRIPTOR_DIM)
    arr = _downsample(arr, TARGET_FRAMES)
    best_id, confidence, raw_score, verdict, dtw_x, dtw_y = find_best_match(arr)
    analog_hole_confidence = _compute_analog_hole_confidence_from_signature(arr)
    return {
        "match_id": best_id,
        "confidence": confidence,
        "analog_hole_confidence": analog_hole_confidence,
        "analog_hole_likely": analog_hole_confidence >= ANALOG_HOLE_THRESHOLD,
        "raw_score": raw_score,
        "verdict": verdict,
        "dtw_path_x": dtw_x,
        "dtw_path_y": dtw_y,
    }


@app.post("/verify")
async def verify_video(file: UploadFile = File(...)):
    """Upload video → extract signature → compare against reference DB."""
    print("Received file upload")
    video_bytes = await file.read()

    if not video_bytes:
        return {"error": "Empty file"}

    MAX_BYTES = 200 * 1024 * 1024
    if len(video_bytes) > MAX_BYTES:
        return {"error": f"File too large (limit {MAX_BYTES // (1024 * 1024)} MB)"}

    query_sig, analog_hole_confidence, analog_hole_metrics = extract_signature(video_bytes)
    best_id, confidence, raw_score, verdict, dtw_x, dtw_y = find_best_match(query_sig)

    if best_id == -1:
        return {"error": "No reference database loaded"}

    query_signal_norm = np.linalg.norm(query_sig, axis=1).tolist()
    ref_signal_norm = np.linalg.norm(demoval[best_id], axis=1).tolist()
    
    gemini_insights = None
    if os.environ.get("USE_GEMINI", "").lower() == "true":
        client = GeminiClient()
        match_metadata = {
            "match_id": best_id,
            "confidence": confidence,
            "analog_hole_confidence": analog_hole_confidence,
            "raw_score": raw_score,
            "verdict": verdict,
            "file_name": file.filename
        }
        # Pass the 1D motion energy (magnitude) arrays for easier LLM interpretation
        gemini_insights = client.generate_forensic_report(
            query_sig=query_signal_norm,
            ref_sig=ref_signal_norm,
            match_metadata=match_metadata
        )

    return {
        "file_name": file.filename,
        "match_id": best_id,
        "confidence": confidence,
        "analog_hole_confidence": analog_hole_confidence,
        "analog_hole_likely": analog_hole_confidence >= ANALOG_HOLE_THRESHOLD,
        "analog_hole_metrics": analog_hole_metrics,
        "raw_score": raw_score,
        "verdict": "MATCH" if raw_score < MATCH_THRESHOLD else "NO MATCH",
        "query_signal": query_signal_norm,
        "ref_signal": ref_signal_norm,
        "gemini_insights": gemini_insights,
        "dtw_path_x": dtw_x,
        "dtw_path_y": dtw_y,
    }
