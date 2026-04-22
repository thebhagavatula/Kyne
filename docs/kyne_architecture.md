# Kyne (Kinetic Perceptual Entropy)
## System Architecture & Pipeline Documentation

**Branch:** `dev`
**Role Context:** Core Algorithm Engine (Layers 1–3)

---

## 1. System Overview

**Kyne** (codenamed *Project Codex / KPE*) is a motion-signature-based video fingerprinting system. Its primary purpose is to detect pirated sports broadcasts and video content. Unlike traditional frame-hashing or audio-fingerprinting, KPE tracks the underlying **motion physics** (camera pans, zooms, player movements). This makes the fingerprinting robust against common piracy obfuscation techniques such as:
- Heavy cropping or zooming
- Recoloring or brightness alterations
- "Cam" recordings (filming a screen with a phone)

The system is designed as a **5-Layer Pipeline**, though the current `dev` branch only contains the completed implementations for Layers 1, 2, and 3.

---

## 2. Pipeline Architecture & Data Flow

```mermaid
graph TD
    A[Raw Video/MP4] -->|Layer 1: Ingest| B(VideoPreprocessor)
    B -->|Grayscale 320x240 Arrays| C(SparseOpticalFlowExtractor)
    C -->|Layer 2: Extract| D[Optical Flow Tracking]
    D -->|Raw 10-dim Descriptors| E[build_signature]
    E -->|Smoothing & L2 Normalisation| F(S[t] Signature Array)
    F -->|Layer 3: Store| G(SignatureStore)
    G --> H[(.npy Storage)]
    G --> I[(.json Manifest)]
    G -.->|to_json_payload| J[Layer 4: Verify API *Pending*]
```

### Layer 1: Ingest (`kpe.extract.preprocessor`)
**Goal:** Standardize incoming video sources to ensure deterministic tracking behavior and control computational costs.
- **Component:** `VideoPreprocessor`
- **Mechanism:**
  - Ingests raw video files (`.mp4`, etc.) or in-memory array streams.
  - Forces conversion to grayscale to strip color dependencies.
  - Downscales all frames to a strict `320x240` resolution (configurable).
  - Optimises throughput by applying a `frame_skip` algorithm (e.g., analyzing every 2nd frame) to stay within the 30fps/33ms real-time processing budget.

### Layer 2: Extract (`kpe.extract.optical_flow`)
**Goal:** The technical "moat" of the system. Converts spatial frames into a temporal physics stream `S(t)`.
- **Component:** `SparseOpticalFlowExtractor`
- **Mechanism:**
  1. **Corner Detection:** Uses the Shi-Tomasi `goodFeaturesToTrack` algorithm to locate high-contrast corners in the frame (default 200 points).
  2. **Propagation:** Uses the Lucas-Kanade (`calcOpticalFlowPyrLK`) sparse tracking algorithm to find where those exact corners moved in the subsequent frame.
  3. **Descriptor Construction:** The motion vectors of the tracked points are binned into a **10-dimensional descriptor**:
     - 4 bins for magnitude (how fast objects moved)
     - 4 bins for angle/direction (0–360°)
     - 2 values for global mean displacement (X and Y camera panning)
  4. **Post-Processing (`build_signature`):**
     - Interpolates over minor tracking dropouts (< 5 frames) using numpy linear interpolation.
     - Flattens high-frequency tracking jitter using a temporal median filter (window size = 3).
     - Normalizes the vectors using L2 normalisation so that matching focuses on the *shape* of the motion rather than the absolute physical speed.

### Layer 3: Store (`kpe.store.signature_store`)
**Goal:** Safely persist the signatures and expose them to the downstream matching API.
- **Component:** `SignatureStore`
- **Mechanism:**
  - Saves the resulting `S(t)` numpy array as a highly compressed `.npy` file.
  - Generates a sidecar `.json` manifest containing metadata (frame count, dimension limits, drop ratios, timestamps).
  - Provides a `to_json_payload()` serializer that unpacks the numpy array into a nested list suitable for HTTP transport to the FastAPI microservice.

---

## 3. Current Integrations & Dependencies

As of the current `dev` branch, Kyne has zero external microservice dependencies and runs entirely locally. The core technology stack consists of:

- **OpenCV (`cv2`)**: Used strictly for spatial transformations (resizing, grayscaling) and the core C++ implementations of Shi-Tomasi and Lucas-Kanade.
- **NumPy (`np`)**: Used for all heavy matrix manipulations, vector binning, interpolation, filtering, and standard `.npy` serialization.
- **PyTest**: Powers the parametric integration testing suite to validate physics boundaries.

> [!WARNING]
> **Pending Integrations (Layers 4 & 5)**
> The `dev` branch does *not* currently integrate the `dtaidistance` package, the FastAPI backend, or the Streamlit dashboard. The current architectural boundary ends at the JSON payload emission in `SignatureStore`.

---

## 4. System Operation

To run the full end-to-end extraction natively in the `dev` branch, the system utilizes a CLI entrypoint wrapper:

```bash
python scripts/extract_kpe.py --input path/to/video.mp4 --output_dir output/
```

This single command triggers the full Layer 1 -> Layer 3 cascade, outputting the finalized fingerprint ready for future verification.
