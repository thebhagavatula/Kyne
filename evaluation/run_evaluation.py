import requests
import csv
import sys
from pathlib import Path

url = "http://localhost:8000/verify"

results = []

with open("dataset/pairs.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        clip_a    = row["clip_a"]
        clip_b    = row["clip_b"]
        expected  = int(row["expected"])

        # verify clip_a against the reference DB
        try:
            with open(clip_a, "rb") as f:
                response = requests.post(url, files={"file": f}, timeout=60)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"Error processing {clip_a}: {e}")
            continue

        raw_score  = result.get("raw_score", None)
        confidence = result.get("confidence", 0)
        verdict    = result.get("verdict", "NO MATCH")
        match_id   = result.get("match_id", -1)

        results.append({
            "clip_a":     clip_a,
            "clip_b":     clip_b,
            "expected":   expected,
            "raw_score":  raw_score,
            "confidence": confidence,
            "verdict":    verdict,
            "match_id":   match_id,
        })

        print(f"clip_a={Path(clip_a).name:30s}  expected={expected}  "
              f"raw_score={raw_score:.4f}  confidence={confidence:.4f}  verdict={verdict}")

# ── score distribution ──────────────────────────────────────────────────────
match_scores    = [r["raw_score"] for r in results if r["expected"] == 1 and r["raw_score"] is not None]
nonmatch_scores = [r["raw_score"] for r in results if r["expected"] == 0 and r["raw_score"] is not None]

print("\n========== SCORE DISTRIBUTION ==========")
if match_scores:
    print(f"MATCH    scores  — min={min(match_scores):.4f}  max={max(match_scores):.4f}  "
          f"mean={sum(match_scores)/len(match_scores):.4f}")
if nonmatch_scores:
    print(f"NO MATCH scores  — min={min(nonmatch_scores):.4f}  max={max(nonmatch_scores):.4f}  "
          f"mean={sum(nonmatch_scores)/len(nonmatch_scores):.4f}")

print("\n  → Set MATCH_THRESHOLD between the two score ranges above")
print("    e.g. midpoint =", round(
    (max(match_scores or [0]) + min(nonmatch_scores or [9999])) / 2, 2
))

# ── threshold sweep ──────────────────────────────────────────────────────────
print("\n========== THRESHOLD SWEEP ==========")
all_scores = [(r["raw_score"], r["expected"]) for r in results if r["raw_score"] is not None]
candidate_thresholds = sorted(set(s for s, _ in all_scores))

best_acc, best_thresh = 0, None
for thresh in candidate_thresholds:
    correct = sum(
        1 for score, exp in all_scores
        if (1 if score < thresh else 0) == exp
    )
    acc = correct / len(all_scores)
    print(f"  threshold={thresh:.4f}  accuracy={acc:.2f}")
    if acc > best_acc:
        best_acc, best_thresh = acc, thresh

print(f"\n  → Best threshold: {best_thresh:.4f}  (accuracy={best_acc:.2f})")
print(f"    Set MATCH_THRESHOLD = {best_thresh:.4f} in api/main.py")