# GENROGRAM — Dataset & Preprocessing Specifications

## Dataset Overview

- **Source:** GTZAN Genre Collection
- **Total Tracks:** 1,000 audio files (100 per genre)
- **Genres:** Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
- **Track Duration:** 30 seconds
- **Audio Specs:** 22,050 Hz sampling rate, 16-bit mono PCM WAV format

## Train / Validation / Test Split Strategy

- **Track-Level Split:** 70% Train (700 tracks), 15% Validation (150 tracks), 15% Test (150 tracks).
- **Leakage Prevention:** Splitting is performed exclusively at the original track level before segmentation.
- **Reproducibility:** Seeded random state (`seed=42`) stored in `data/split_manifest.json`.

## Preprocessing Parameters

- `sample_rate`: 22,050 Hz
- `n_mels`: 128
- `n_fft`: 2,048
- `hop_length`: 512
- `segment_duration`: 3.0s (~130 time frames per spectrogram)
