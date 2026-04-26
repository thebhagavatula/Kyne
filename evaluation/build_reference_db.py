from api.main import extract_signature

video_files = [
    "dataset/clips/video_1.mp4",
    "dataset/clips/video_2.mp4",
    "dataset/clips/video_3.mp4",
    "dataset/clips/video_4.mp4",
    "dataset/clips/video_5.mp4"
]

reference_signatures = []

for path in video_files:
    with open(path, "rb") as f:
        video_bytes = f.read()

    signature = extract_signature(video_bytes)
    reference_signatures.append(signature)

import json

with open("dataset/reference_signatures.json", "w") as f:
    json.dump(reference_signatures, f)

print("Reference DB saved")