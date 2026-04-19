# Kyne
> Motion-signature video fingerprinting pipeline for piracy detection in broadcasts
[![Google Solution Challenge 2025](https://img.shields.io/badge/Google-Solution%20Challenge-blue)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)]()

## Problem
Pirated sports streams evade hash/watermark detection via cropping, recolouring,
and screen-recording. KPE detects copies by their *motion signature* — invariant
to these transforms.

## How It Works — 5-Layer Pipeline

| Layer    | Component                        | Tech                          |
|----------|----------------------------------|-------------------------------|
| Ingest   | Frame sampler                    | OpenCV, 320×240 resize        |
| Extract  | Sparse optical flow → descriptor | Lucas-Kanade, 10-dim vector   |
| Store    | Signature index                  | NumPy .npy / JSON manifest    |
| Verify   | DTW matching + confidence score  | dtaidistance                  |
| Demo     | Dashboard                        | Streamlit                     |

## Quick Start
\`\`\`bash
pip install -r requirements.txt
uvicorn api.main:app --reload          # Start verification API
streamlit run kpe/demo/dashboard.py    # Launch demo dashboard
\`\`\`

## API
POST /verify — submit a clip, receive DTW distance + MATCH/NO MATCH verdict

## Evaluation Results
| Test Set         | Precision | Recall |
|------------------|-----------|--------|
| Genuine (n=10)   | TBD       | TBD    |
| Non-match (n=10) | TBD       | TBD    |
| Distorted (n=5)  | TBD       | TBD    |
