from fastapi import FastAPI
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

@app.get("/")
def read_root():
    return {"message": "Sab Khairiyat"}

@app.post("/match")
def match_vid(inputData:InputVid):
    best_score = float("inf")
    best_id = -1

    for i, ref in enumerate(demoval):
        score = dtw.distance(inputData.query_signature, ref)

        if score < best_score:
            best_score = score
            best_id = i

    confidence = 1/(1+best_score)

    return {
        "match_id": best_id,
        "confidence": confidence,
        "verdict": "Match" if confidence > 0.5 else "No matches"
    }