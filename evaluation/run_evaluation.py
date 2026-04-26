import requests
import csv

url = "http://localhost:8000/verify"

correct = 0
total = 0

false_positive = 0
false_negative = 0

with open("dataset/pairs.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        clip = row["clip_a"]   # only using clip_a
        expected = int(row["expected"])

        try:
            with open(clip, "rb") as f:
                files = {
                    "file": f   # IMPORTANT: single input
                }

                response = requests.post(url, files=files)

                print("Status:", response.status_code)

                result = response.json()
                print("Response:", result)

        except Exception as e:
            print(f"Error processing {clip}: {e}")
            continue

        # Safely extract values
        confidence = result.get("confidence", 0)
        verdict = result.get("verdict", "NO MATCH")

        predicted = 1 if verdict == "MATCH" else 0

        print(f"\nClip: {clip}")
        print(f"Expected: {expected}, Predicted: {predicted}, Confidence: {confidence}\n")

        # Metrics
        if predicted == expected:
            correct += 1

        if predicted == 1 and expected == 0:
            false_positive += 1

        if predicted == 0 and expected == 1:
            false_negative += 1

        total += 1

# Final metrics
if total > 0:
    accuracy = correct / total
else:
    accuracy = 0

print("========== FINAL RESULTS ==========")
print(f"Accuracy: {accuracy:.2f}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")