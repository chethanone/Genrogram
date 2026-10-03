# GENROGRAM

> **Academic Project:** Develop a Music Genre Classification System Using Spectrogram and CNN

GENROGRAM is a full-stack music genre classification application that converts uploaded audio into log-Mel spectrograms and classifies the track with a PyTorch Custom Convolutional Neural Network (CustomCNN). The completed system is deployed as a single Vercel application with a Next.js frontend and FastAPI inference backend.

The current production system supports **20 music genres**, provides track-level predictions and confidence scores, generates spectrogram visualizations, and provides Grad-CAM explainability for the model's prediction.

---

## Live Demo

**Production:** https://genrogram.vercel.app

GENROGRAM is deployed as a production Vercel Services application. The frontend and FastAPI backend are served from the same production origin, with backend API routes available under `/api`.

---

## Overview

GENROGRAM combines digital audio signal processing, supervised deep learning, and an interactive web interface.

### Processing pipeline

```text
Audio Upload
    ↓
Audio Validation
    ↓
Mono / 22.05 kHz Processing
    ↓
Onset Detection + 3-Second Segmentation
    ↓
128-Band Log-Mel Spectrogram
    ↓
CustomCNN
    ↓
20-Class Probability Distribution
    ↓
Predicted Genre + Confidence
```

For a track containing multiple 3-second segments, the model predicts each segment and the probability vectors are averaged to produce the track-level prediction.

---

## Current Production Model

**Architecture:** CustomCNN  
**Parameters:** 393,940  
**Classes:** 20  
**Model version:** `20-class-production-v1`

### Supported genres

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock
- Amapiano
- Hyperpop
- K-Pop
- Phonk
- Techno
- Bollywood
- Desi Hip-Hop
- Haryanvi
- I-Pop
- Punjabi Pop

### Audio and spectrogram configuration

| Setting | Production value |
|---|---:|
| Sample rate | 22,050 Hz |
| Segment duration | 3 seconds |
| Mel bands | 128 |
| FFT size | 2,048 |
| Hop length | 512 |
| Spectrogram | Log-Mel |
| dB reference | Maximum |
| Normalization | Per-segment min-max |

Training-time augmentation uses frequency masking, time masking, and controlled Gaussian noise. No augmentation is applied during production inference.

---

## Evaluation

The production checkpoint was evaluated on an untouched held-out test set of **323 tracks across 20 genres**.

### Track-level results

| Metric | Result |
|---|---:|
| Accuracy | **75.54%** |
| Macro Precision | **78.72%** |
| Macro Recall | **75.68%** |
| Macro F1 | **75.58%** |

### Segment-level results

| Metric | Result |
|---|---:|
| Accuracy | **65.56%** |
| Macro Precision | **68.30%** |
| Macro Recall | **65.47%** |
| Macro F1 | **65.55%** |

Track-level evaluation is the primary production metric because the application ultimately classifies an uploaded song rather than an isolated spectrogram segment.

---

## Dataset and Training

The current 20-class training corpus contains **2,143 tracks**:

- Train: 1,499 tracks
- Validation: 321 tracks
- Test: 323 tracks

The training corpus combines the project's original GTZAN material with additional genre data used to expand the system to 20 classes.

The data is split at the **track level**, so segments from the same source track are kept within the same split. This prevents segments from one song appearing across training, validation, and test sets.

Raw datasets and training caches are intentionally not part of the deployment runtime.

---

## Application Features

### Classify

Upload an audio file and receive:

- Predicted genre
- Confidence score
- Top predictions
- Audio playback
- Track analysis

Supported upload formats:

- WAV
- MP3
- FLAC
- OGG
- M4A

### Spectrogram Lab

View the generated log-Mel spectrogram and inspect the production preprocessing configuration used by the system.

### Explainability

Generate a Grad-CAM visualization showing the regions of the spectrogram that contributed to the selected prediction.

### Model

View the production model architecture, preprocessing configuration, training configuration, and architecture comparison information.

### Performance

View the held-out evaluation metrics for the production model.

### History

Classification history is stored locally in the browser using `localStorage`. No Firebase authentication or Firestore database is used.

---

## Technology Stack

### Frontend

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Framer Motion
- Recharts
- Lucide React

### Backend

- FastAPI
- Uvicorn
- Pydantic
- Python

### Machine Learning

- PyTorch
- Librosa
- NumPy
- SoundFile
- Matplotlib

The production backend runs the trained CustomCNN checkpoint directly for inference.

---

## Project Structure

```text
Genrogram/
├── backend/
│   ├── main.py                   # FastAPI application
│   ├── model_service.py          # Production model inference
│   ├── audio_service.py          # Audio and spectrogram processing
│   ├── explainability_service.py # Grad-CAM generation
│   └── schemas.py                # API response schemas
├── configs/
├── frontend/
│   └── app/                      # Next.js application
├── ml/
│   └── custom_cnn.py             # CustomCNN architecture
├── models/
│   └── production/
│       ├── gengrogram_custom_cnn_20class_production.pt
│       └── production_config.json
├── results/
├── scripts/
├── tests/
├── docs/
├── requirements.txt
└── vercel.json
```

Training datasets, caches, virtual environments, build output, and other development-only artifacts are excluded from deployment where appropriate.

---

## Local Development

### Frontend

From the `frontend` directory:

```bash
npm install
npm run dev
```

The Next.js development server normally runs at:

```text
http://localhost:3000
```

Set the backend URL through:

```text
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api
```

### Backend

From the project root, install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then start FastAPI:

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend normally runs at:

```text
http://127.0.0.1:8000
```

The frontend communicates with the backend through `NEXT_PUBLIC_API_URL`.

---

## API

The production backend exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend and model health |
| `/model/info` | GET | Production model information |
| `/predict` | POST | Predict the genre of an uploaded track |
| `/spectrogram` | POST | Generate a spectrogram image |
| `/explainability` | POST | Generate Grad-CAM visualization |
| `/evaluation/summary` | GET | Evaluation summary |
| `/evaluation/confusion-matrix` | GET | Evaluation confusion matrix |

---

## Deployment

GENROGRAM is configured for a **single Vercel project using Vercel Services**:

1. **Frontend service** — Next.js using `frontend/`.
2. **Backend service** — FastAPI using `backend.main:app`.

The deployment exposes the FastAPI service under the same origin at `/api`, while all other routes are handled by the Next.js frontend. This keeps the production frontend and backend on the same origin and avoids a separate public backend URL.

The production model, ML modules, configuration, evaluation JSON, and confusion matrix are included in the backend service bundle.

The current client-side upload limit is 4 MB to remain within the deployment request-size constraint.

---

## Production Artifact

The frozen production checkpoint is:

```text
models/production/gengrogram_custom_cnn_20class_production.pt
```

Its corresponding configuration is:

```text
models/production/production_config.json
```

The production configuration records the model version, class mapping, preprocessing parameters, augmentation settings, and held-out evaluation metrics.

---

## Project Status

**Status: Complete — production system deployed.**

GENROGRAM currently has:

- A trained 20-class production CustomCNN
- A FastAPI inference backend
- A responsive Next.js frontend
- Spectrogram visualization
- Grad-CAM explainability
- Local classification history
- Production evaluation metrics
- Vercel deployment configuration

The project is complete and deployed as a live production system. The current release includes the frozen 20-class model, inference backend, responsive interface, explainability, evaluation views, local history, and production deployment configuration.

---

## Academic Context

GENROGRAM was developed as an academic machine learning project demonstrating the use of:

- Digital audio signal processing
- Mel-spectrogram representation
- Convolutional neural networks
- Supervised multi-class classification
- Track-level evaluation
- Model explainability with Grad-CAM
- Full-stack ML application deployment
