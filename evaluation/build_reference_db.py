import json
import sys
from pathlib import Path

# make sure project root is on path regardless of where script is invoked
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.main import extract_signature

video_files = [
    "dataset/clips/video_1.mp4",
    "dataset/clips/video_2.mp4",
    "dataset/clips/video_3.mp4",
    "dataset/clips/video_4.mp4",
    "dataset/clips/video_5.mp4",
]

reference_signatures = []

for path in video_files:
    try:
        print(f"Processing {path}...")
        with open(path, "rb") as f:
            video_bytes = f.read()

        sig = extract_signature(video_bytes)  # returns np.ndarray (T, 10)

        # convert to nested list so json.dump can serialise it
        reference_signatures.append(sig.tolist())
        print(f"  Done — signature shape: {sig.shape}")

    except FileNotFoundError:
        print(f"  Skipping {path} — file not found")
    except Exception as e:
        print(f"  Error processing {path}: {e}")

if not reference_signatures:
    print("No signatures were generated. Aborting — file not written.")
    sys.exit(1)

out_path = Path("dataset/reference_signatures.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(reference_signatures, f)

print(f"\nReference DB saved — {len(reference_signatures)} signatures → {out_path}")