# GENROGRAM

> **Academic Title:** Develop a Music Genre Classification System Using Spectrogram and CNN  
> **Architecture:** Audio → Resampling (22.05 kHz) → Mel Spectrogram → PyTorch Custom CNN → 10-Genre Softmax

---

## 1. Project Overview

GENROGRAM is a full-stack, reproducible music genre classification system powered by deep learning and spectral signal processing. It extracts Mel-spectrograms from audio signals and classifies them using a multi-layer Convolutional Neural Network (CNN).

### Key Features
- **Audio Preprocessing:** Resampling (22,050 Hz), mono conversion, dynamic range normalization, track-level segmentation, and log-Mel spectrogram generation (128 mel bins).
- **Custom CNN Model:** PyTorch implementation with batch normalization, global average pooling, dropout, and Softmax output.
- **Explainability:** Grad-CAM activation mapping to visualize model focus regions on Mel-spectrograms.
- **Interactive UI:** Next.js application featuring Spectrogram Lab, real-time audio player, waveform renderer, model diagnostics, and performance metrics.
- **Authentication & Persistence:** Firebase Auth (Email/Password, Google OAuth) and Firestore database for classification history.

---

## 2. Dataset & Split Policy

- **Primary Dataset:** GTZAN Genre Collection (1,000 30-second WAV tracks across 10 genres).
- **Split Strategy:** Track-level deterministic split (70% Train, 15% Validation, 15% Test) with `seed=42`. Segment-level splitting is strictly prohibited to avoid data leakage.

---

## 3. Technology Stack

- **ML Pipeline:** PyTorch, Librosa, NumPy, Scikit-Learn
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Next.js (TypeScript, Tailwind CSS, Framer Motion, Recharts, Lucide React)
- **Database & Auth:** Firebase Auth, Cloud Firestore

---

## 4. Documentation Index

- [Architecture Guide](docs/architecture.md)
- [Dataset Specifications](docs/dataset.md)
- [Training & Safety Protocols](docs/training.md)
- [Evaluation & Metrics](docs/evaluation.md)
- [Deployment Guide](docs/deployment.md)
- [Authentication & Security](docs/authentication.md)
