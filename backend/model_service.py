from pathlib import Path
import json

import torch

from ml.custom_cnn import CustomCNN
from ml.audio_processor import AudioProcessor


class ModelService:
    """
    Production model service for GENGROGRAM.

    Loads the frozen 20-class Custom CNN and performs
    track-level genre prediction using the exact inference
    preprocessing used during model training.
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent

        # --------------------------------------------------------
        # Production artifacts
        # --------------------------------------------------------

        self.model_path = (
            self.project_root
            / "models"
            / "production"
            / "gengrogram_custom_cnn_20class_production.pt"
        )

        self.production_config_path = (
            self.project_root
            / "models"
            / "production"
            / "production_config.json"
        )

        # Existing audio processor configuration.
        #
        # This must contain the same preprocessing parameters
        # used during training:
        #
        #   sample_rate = 22050
        #   segment_seconds = 3.0
        #   n_mels = 128
        #   n_fft = 2048
        #   hop_length = 512
        #
        self.config_path = (
            self.project_root
            / "configs"
            / "default_config.json"
        )

        # --------------------------------------------------------
        # Device
        # --------------------------------------------------------

        # Web deployment currently uses CPU inference.
        # This keeps the backend portable and avoids requiring
        # CUDA on the local web server.
        self.device = torch.device("cpu")

        # --------------------------------------------------------
        # Production configuration
        # --------------------------------------------------------

        self.production_config = self._load_production_config()

        # --------------------------------------------------------
        # Class names
        # --------------------------------------------------------

        self.class_names = self._load_class_names()

        if len(self.class_names) != 20:
            raise RuntimeError(
                "Production model requires exactly 20 classes, "
                f"but {len(self.class_names)} were found."
            )

        # --------------------------------------------------------
        # Audio preprocessing
        # --------------------------------------------------------

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Audio configuration not found: {self.config_path}"
            )

        self.audio_processor = AudioProcessor(
            config_path=str(self.config_path)
        )

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        self.model = CustomCNN(
            num_classes=len(self.class_names)
        )

        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        self.model_name = "CustomCNN"
        self.model_version = "20-class-production-v1"

        self.sample_rate = 22050
        self.segment_seconds = 3.0
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def _load_production_config(self):
        """Load the frozen production configuration."""

        if not self.production_config_path.exists():
            raise FileNotFoundError(
                "Production configuration not found:\n"
                f"{self.production_config_path}"
            )

        with open(
            self.production_config_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    # ============================================================
    # CLASS MAPPING
    # ============================================================

    def _load_class_names(self):
        """
        Load the 20 production class names.

        The frozen production configuration is the authoritative
        source for the production class order.
        """

        config_classes = self.production_config.get(
            "class_names"
        )

        if isinstance(config_classes, list):
            return [str(name) for name in config_classes]

        # Fallback to the production checkpoint metadata.
        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            checkpoint_classes = checkpoint.get(
                "class_names"
            )

            if isinstance(checkpoint_classes, list):
                return [
                    str(name)
                    for name in checkpoint_classes
                ]

        # Final explicit production mapping.
        return [
            "Blues",
            "Classical",
            "Country",
            "Disco",
            "Hip-Hop",
            "Jazz",
            "Metal",
            "Pop",
            "Reggae",
            "Rock",
            "Amapiano",
            "Hyperpop",
            "K-Pop",
            "Phonk",
            "Techno",
            "Bollywood",
            "Desi Hip-Hop",
            "Haryanvi",
            "I-Pop",
            "Punjabi Pop",
        ]

    # ============================================================
    # CHECKPOINT
    # ============================================================

    def _load_checkpoint(self):
        """Load and validate the frozen production checkpoint."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Production model checkpoint not found:\n"
                f"{self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # --------------------------------------------------------
        # Extract state dictionary
        # --------------------------------------------------------

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint[
                    "model_state_dict"
                ]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint[
                    "state_dict"
                ]

            else:
                # Possible raw state_dict.
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        try:
            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

        except RuntimeError as exc:
            raise RuntimeError(
                "The production checkpoint does not match "
                "the expected 20-class CustomCNN architecture.\n\n"
                f"Checkpoint: {self.model_path}\n\n"
                f"Original error:\n{exc}"
            ) from exc

    # ============================================================
    # PREDICTION
    # ============================================================

    @torch.no_grad()
    def predict_file(self, file_path: str):
        """
        Predict the genre of a complete audio track.

        Processing:
            audio
              ↓
            3-second segments
              ↓
            log-Mel features
              ↓
            Custom CNN
              ↓
            segment probabilities
              ↓
            mean probability aggregation
              ↓
            track prediction
        """

        file_path = str(file_path)

        # --------------------------------------------------------
        # Feature extraction
        # --------------------------------------------------------

        tensor = self.audio_processor.process_file_to_tensor(
            file_path
        )

        if tensor.ndim != 4:
            raise RuntimeError(
                "Unexpected audio tensor shape: "
                f"{tuple(tensor.shape)}"
            )

        if tensor.shape[1:] != (1, 128, 130):
            raise RuntimeError(
                "Unexpected production feature shape: "
                f"{tuple(tensor.shape)}. "
                "Expected [N, 1, 128, 130]."
            )

        tensor = tensor.to(self.device)

        # --------------------------------------------------------
        # Model inference
        # --------------------------------------------------------

        logits = self.model(tensor)

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Model produced non-finite logits."
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # --------------------------------------------------------
        # Track-level aggregation
        # --------------------------------------------------------

        track_probabilities = probabilities.mean(
            dim=0
        )

        predicted_index = int(
            torch.argmax(
                track_probabilities
            ).item()
        )

        predicted_genre = self.class_names[
            predicted_index
        ]

        # --------------------------------------------------------
        # Probability dictionary
        # --------------------------------------------------------

        probability_values = {
            self.class_names[i]: float(
                track_probabilities[i].item()
            )
            for i in range(
                len(self.class_names)
            )
        }

        # --------------------------------------------------------
        # Result
        # --------------------------------------------------------

        return {
            "genre": predicted_genre,

            "confidence": float(
                track_probabilities[
                    predicted_index
                ].item()
            ),

            "probabilities": probability_values,

            "segment_count": int(
                tensor.shape[0]
            ),

            "class_index": predicted_index,

            "model": self.model_name,

            "model_version": self.model_version,
        }

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    def get_model_info(self):
        """Return metadata used by the frontend/model page."""

        return {
            "name": self.model_name,
            "version": self.model_version,
            "architecture": "Custom CNN",
            "parameters": 393940,
            "num_classes": len(
                self.class_names
            ),
            "classes": self.class_names,

            "sample_rate": self.sample_rate,
            "segment_seconds": self.segment_seconds,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,

            "feature_shape": [
                1,
                128,
                130,
            ],

            "test_track_accuracy": 0.7554,
            "test_track_macro_f1": 0.7558,

            "device": str(self.device),

            "checkpoint": str(
                self.model_path
            ),
        }


# ================================================================
# SINGLETON
# ================================================================

# The model is loaded once when the backend starts.
# Requests reuse the same model instance.
model_service = ModelService()