from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dtaidistance import dtw

app = FastAPI()

demoval = [
    [0.1, 0.4, 0.6, 0.8],
    [0.9, 0.2, 0.3, 0.1]
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


def find_best_match(query_signature):
    best_score = float("inf")
    best_id = -1

    for i, ref in enumerate(demoval):
        score = dtw.distance(query_signature, ref)

        if score < best_score:
            best_score = score
            best_id = i

    confidence = 1 / (1 + best_score)

    return best_id, confidence

# /match is jus for debugging
@app.post("/match")
def match_vid(inputData: InputVid):
    best_id, confidence = find_best_match(inputData.query_signature)

    return {
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH"
    }

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
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "MATCH" if confidence > 0.5 else "NO MATCH"
    }