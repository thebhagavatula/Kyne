from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dtaidistance import dtw

app = FastAPI()

demoval = [
    [0.1 * i for i in range(20)],
    [0.9 * i for i in range(20)]
]

class InputVid(BaseModel):
    query_signature: List[float]
    video_id: Optional[str] = None

def extract_signature(video_bytes):
    # this is mock for now, this will call the extracting function
    import random
    return [random.random() for _ in range(20)]

@app.get("/")
def read_root():
    return {"message": "Sab Khairiyat"}

# the input video can be a part of da reference video so we will need windows of traversal
def sliding_dtw(query, ref):
    window_size = len(query)
    best = float("inf")

    if len(ref) < window_size:
        return dtw.distance(query, ref)
    
    for i in range(len(ref) - window_size + 1):
        window = ref[i:i + window_size]
        score = dtw.distance(query, window)
        best = min(best, score)

    return best

# Confidence formula needs to be changed
def find_best_match(query_signature):
    best_score = float("inf")
    best_id = -1

    for i, ref in enumerate(demoval):
        score = sliding_dtw(query_signature, ref)

        if score < best_score:
            best_score = score
            best_id = i

    confidence = 1 / (1 + best_score)
    return best_id, confidence

# /match is jus for summa
@app.post("/match")
def match_vid(inputData: InputVid):
    best_id, confidence = find_best_match(inputData.query_signature)

    return {
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH"
    }

# route at which upload happens and find_best_match is called
@app.post("/verify")
async def verify_video(file: UploadFile = File(...)):
    video_bytes = await file.read()

    if not video_bytes:
        return {"error": "Empty file"}

    if len(video_bytes) > 5_000_000:
        return {"error": "File too large"}

    query_signature = extract_signature(video_bytes)

    best_id, confidence = find_best_match(query_signature)

    return {
        "file_name": file.filename,
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH",
        "query_signal": query_signature,
        "ref_signal": demoval[best_id]
    }