# GENROGRAM — System Architecture

## Overview

```text
Audio File (.wav, .mp3)
       ↓
[Audio Processor] → Resample (22.05 kHz) → Mono → Segment (3.0s) → Log-Mel Spectrogram (128x130)
       ↓
[Custom CNN Model] → Conv2D/BN/ReLU/MaxPool x3 → Conv2D/BN/ReLU/GAP → Dropout → Dense(10)
       ↓
[FastAPI Backend] → `/predict` API Endpoint + Grad-CAM Heatmap Generation
       ↓
[Next.js Frontend] → Retro-futuristic UI (Spectrogram Lab, Prediction Cards, Performance Charts)
```

## Component Architecture

1. **ML Pipeline (`/ml`)**: Pure PyTorch dataset definitions, audio transform routines, CNN architecture models, and Grad-CAM explainability modules.
2. **Backend API (`/backend`)**: FastAPI REST endpoints delivering predictions, spectrogram computations, evaluation metrics, and user classification history.
3. **Frontend UI (`/frontend`)**: Next.js App Router client with responsive design, Framer Motion animations, Recharts metrics visualization, and Firebase integration.
