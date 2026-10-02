import json
import numpy as np
import librosa
import torch
from pathlib import Path

class AudioProcessor:
    def __init__(self, config_path: str = "configs/default_config.json"):
        cfg_file = Path(config_path)
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                config = json.load(f)
            audio_cfg = config.get("audio", {})
        else:
            audio_cfg = {}

        self.sample_rate = audio_cfg.get("sample_rate", 22050)
        self.n_mels = audio_cfg.get("n_mels", 128)
        self.n_fft = audio_cfg.get("n_fft", 2048)
        self.hop_length = audio_cfg.get("hop_length", 512)
        self.segment_duration = audio_cfg.get("segment_duration", 3.0)
        self.segment_samples = int(self.segment_duration * self.sample_rate)

    def load_and_preprocess_raw_audio(self, file_path: str) -> np.ndarray:
        """Loads audio, resamples to target SR, converts to mono, normalizes amplitude."""
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        # Peak normalization
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def extract_segments(self, y: np.ndarray) -> list[np.ndarray]:
        """Splits full audio track into 3.0 second non-overlapping segments."""
        total_samples = len(y)
        segments = []
        
        # Extract full duration 3s windows
        for start in range(0, total_samples - self.segment_samples + 1, self.segment_samples):
            seg = y[start : start + self.segment_samples]
            segments.append(seg)
            
        # If total audio is short or last remainder >= 1.5s, pad to 3s
        if not segments or (total_samples % self.segment_samples) >= (self.segment_samples // 2):
            start = max(0, total_samples - self.segment_samples)
            seg = y[start:]
            if len(seg) < self.segment_samples:
                seg = np.pad(seg, (0, self.segment_samples - len(seg)), mode='constant')
            if not segments:
                segments.append(seg)
                
        return segments

    def compute_mel_spectrogram(self, segment: np.ndarray) -> np.ndarray:
        """Computes Log-Mel Spectrogram and normalizes features."""
        # Compute Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=segment,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sample_rate / 2.0
        )
        # Convert to dB scale
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Min-Max Normalization to [0, 1]
        s_min, s_max = S_db.min(), S_db.max()
        if s_max - s_min > 1e-6:
            S_norm = (S_db - s_min) / (s_max - s_min)
        else:
            S_norm = np.zeros_like(S_db)

        return S_norm

    def process_file_to_tensor(self, file_path: str) -> torch.Tensor:
        """Complete pipeline: audio path -> [N, 1, 128, 130] PyTorch Tensor."""
        y = self.load_and_preprocess_raw_audio(file_path)
        segments = self.extract_segments(y)
        
        spectrograms = []
        for seg in segments:
            mel = self.compute_mel_spectrogram(seg)
            spectrograms.append(mel)
            
        # Stack to [N, 1, n_mels, time_steps] tensor
        spec_array = np.array(spectrograms)[:, np.newaxis, :, :]
        tensor = torch.tensor(spec_array, dtype=torch.float32)
        return tensor

if __name__ == "__main__":
    processor = AudioProcessor()
    print(f"AudioProcessor initialized with SR={processor.sample_rate}, Mel Bins={processor.n_mels}, Seg Duration={processor.segment_duration}s")
