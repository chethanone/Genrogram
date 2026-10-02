import io
import tempfile
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.model_service import model_service


class AudioService:
    def __init__(self):
        self.processor = model_service.audio_processor

    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            raise ValueError("Unsupported audio format. Use WAV, MP3, FLAC, OGG, or M4A.")
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(file_bytes)
        temp.close()
        return temp.name

    def detect_music_onset(self, file_path: str):
        """Return audio trimmed at leading silence using an RMS-based threshold."""
        y, sr = librosa.load(file_path, sr=self.processor.sample_rate, mono=True)
        if y.size == 0:
            raise ValueError("Audio contains no samples.")
        y, index = librosa.effects.trim(
            y,
            top_db=35,
            frame_length=2048,
            hop_length=512,
        )
        onset_seconds = float(index[0] / sr)
        return y.astype(np.float32), onset_seconds, sr

    def _three_second_segment_from_onset(self, file_path: str):
        y, onset_seconds, sr = self.detect_music_onset(file_path)
        target = int(round(self.processor.segment_duration * sr))
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        return y, onset_seconds

    def create_spectrogram_image(self, file_path: str):
        """Create the displayed 3-second spectrogram starting at detected music onset."""
        segment, onset_seconds = self._three_second_segment_from_onset(file_path)
        mel = self.processor.compute_mel_spectrogram(segment)

        fig, ax = plt.subplots(figsize=(10, 3.4), dpi=140)
        ax.imshow(
            mel,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            extent=[0, self.processor.segment_duration, 0, self.processor.n_mels],
        )
        ax.set_xlabel("Time from music onset (s)", fontsize=9, labelpad=8)
        ax.set_ylabel("Mel frequency", fontsize=9, labelpad=8)
        ax.tick_params(axis="both", labelsize=8, length=0, colors="#AAA4A0")
        ax.set_title(
            f"Mel Spectrogram | Starts at {onset_seconds:.2f}s",
            fontsize=11,
            fontweight="medium",
            pad=12,
            color="#FFFFFF",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.patch.set_facecolor("#171717")
        ax.set_facecolor("#171717")
        ax.xaxis.label.set_color("#DDD7D3")
        ax.yaxis.label.set_color("#DDD7D3")
        fig.tight_layout(pad=1.4)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue(), onset_seconds


audio_service = AudioService()
