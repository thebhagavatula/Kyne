**Kyne - Kinetic Digital Guardianship**
Motion-based video fingerprinting for sports broadcast piracy detection
Kyne detects unauthorised rebroadcasts of premium sports content by comparing motion signatures instead of pixels. It works even when a pirate has filmed a screen with a phone, cropped the frame, or re-encoded the video at lower resolution, attacks that defeat every existing pixel-based fingerprinting system.

How It Works
    Every broadcast has a unique kinetic signature- the motion of players, cameras, and crowds. Kyne extracts this signature using sparse optical flow (Lucas-Kanade algorithm), represents it as a 10-dimensional descriptor stream called S(t), and matches it against a reference database using Dynamic Time Warping (DTW). The result is a MATCH or NO MATCH verdict with a confidence score, delivered in under 800ms.
    1. suspect clip uploaded
    2. sparse optical flow extraction (Lucas-Kanade, 200 feature points)
    3. 10D motion descriptor per frame → S'(t) signature stream
    4. multivariate sliding DTW vs reference DB (Sakoe-Chiba band)
    5. MATCH / NO MATCH + confidence score + Gemini forensic report

The video clips are stored in the Google Drive: https://drive.google.com/drive/folders/1Z8vzklpkbKTVllfxaf7WQXiuBPW4qzeR?usp=sharing

    Kyne/
├── api/
│   └── main.py                  FastAPI backend with /verify endpoint
├── kpe/
│   ├── extract/
│   │   ├── optical_flow.py      Lucas-Kanade sparse optical flow extractor
│   │   ├── descriptor.py        Motion descriptor computation
│   │   └── exporter.py          Signature export utilities
│   ├── demo/
│   │   └── dashboard.py         Streamlit frontend dashboard
│   ├── store/
│   │   └── signature_store.py   Reference DB management
│   ├── verify/
│   │   └── dtw_matcher.py       DTW matching logic
│   └── ai/
│       └── gemini_client.py     Google Gemini 2.5 Flash integration
├── dataset/
│   ├── clips/                   Reference + test video clips (see Drive link below)
│   ├── reference_signatures.json Pre-extracted reference signatures
│   ├── pairs.csv                Matching pairs for evaluation
│   └── verification_cases.csv  Ground truth for evaluation
├── evaluation/
│   ├── build_reference_db.py    Script to build reference DB from clips
│   └── run_evaluation.py        Batch evaluation script
├── scripts/
│   └── extract_kpe.py           CLI tool for signature extraction
└── requirements.txt

| Component        | Technology                                    |
| ---------------- | --------------------------------------------- |
| Language         | Python 3.13                                   |
| Computer Vision  | OpenCV - Lucas-Kanade optical flow            |
| DTW Matching     | dtaidistance - multivariate, Sakoe-Chiba band |
| API Backend      | FastAPI + Uvicorn                             |
| Dashboard        | Streamlit + Altair                            |
| AI Forensics     | Google Gemini 2.5 Flash                       |
| Data Processing  | NumPy, Pandas                                 |
| Cloud Deployment | Google Cloud Run                              |

SDG Alignment
    Kyne addresses UN Sustainable Development Goal 9 - Industry, Innovation and Infrastructure by protecting the revenue streams that fund premium sports broadcasting infrastructure in emerging markets. Piracy of live sports costs broadcasters an estimated $4.7 billion annually, directly reducing investment in broadcast infrastructure and affordable content access.

Team chaar machli
Project Kyne - Google Solution Challenge 2026