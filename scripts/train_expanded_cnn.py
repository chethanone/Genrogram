"""
GENGROGRAM - Expanded 20-Class CNN Training Pipeline

Dataset:
    3,144 track-level samples
    20 genres
    70/15/15 train/validation/test split

Important:
    - Uses the frozen manifests.
    - Raw audio is never modified.
    - Test set is NOT used during training.
    - Class weights are calculated from TRAIN ONLY.
    - Evaluation is performed at track level.
    - Existing 10-class model is not overwritten.

Training architecture:
    Conv1: 1 -> 32
    Conv2: 32 -> 64
    Conv3: 64 -> 128
    Conv4: 128 -> 256
    Adaptive Average Pool
    Dropout
    Linear: 256 -> 20
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests"

TRAIN_MANIFEST = MANIFEST_ROOT / "train_manifest.csv"
VAL_MANIFEST = MANIFEST_ROOT / "validation_manifest.csv"
TEST_MANIFEST = MANIFEST_ROOT / "test_manifest.csv"

MODEL_ROOT = PROJECT_ROOT / "models" / "expanded_20class"
RESULT_ROOT = PROJECT_ROOT / "results" / "expanded_20class"

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 22050

SEGMENT_SECONDS = 3.0

SEGMENT_SAMPLES = int(
    SAMPLE_RATE * SEGMENT_SECONDS
)

N_MELS = 128

N_FFT = 2048

HOP_LENGTH = 512

BATCH_SIZE = 32

NUM_WORKERS = 0

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 30

PATIENCE = 7

DROPOUT = 0.30


CLASS_NAMES = [
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

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# Utility
# ============================================================

def load_manifest(path: Path):
    import csv

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    return rows


def verify_manifest(rows, name):
    if not rows:
        raise ValueError(
            f"{name} manifest is empty."
        )

    for row in rows:

        genre = row["genre"]

        if genre not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown genre in {name}: {genre}"
            )

        path = Path(
            row["absolute_path"]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing audio file:\n{path}"
            )

    print(
        f"{name:12s}: "
        f"{len(rows):4d} records - verified"
    )


# ============================================================
# Audio preprocessing
# ============================================================

def load_audio(path: str):
    """
    Load mono audio at the fixed training sample rate.
    """

    y, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if y.size == 0:
        raise ValueError(
            f"Empty audio: {path}"
        )

    return y.astype(
        np.float32,
        copy=False,
    )


def pad_or_trim(
    y: np.ndarray,
):
    """
    Convert an audio segment into exactly
    3 seconds.
    """

    if len(y) >= SEGMENT_SAMPLES:
        return y[:SEGMENT_SAMPLES]

    padded = np.zeros(
        SEGMENT_SAMPLES,
        dtype=np.float32,
    )

    padded[:len(y)] = y

    return padded


def make_mel_spectrogram(
    y: np.ndarray,
):
    """
    Produce normalized log-Mel spectrogram.

    Output:
        [1, 128, 130]
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # Per-segment min-max normalization.
    min_value = float(
        np.min(log_mel)
    )

    max_value = float(
        np.max(log_mel)
    )

    denominator = (
        max_value - min_value
    )

    if denominator > 1e-8:

        normalized = (
            log_mel - min_value
        ) / denominator

    else:

        normalized = np.zeros_like(
            log_mel
        )

    tensor = torch.from_numpy(
        normalized.astype(
            np.float32
        )
    ).unsqueeze(0)

    return tensor


def extract_segments(
    y: np.ndarray,
):
    """
    Create non-overlapping 3-second segments.

    A final partial segment is retained and zero-padded.
    """

    if len(y) == 0:
        return []

    segments = []

    start = 0

    while start < len(y):

        segment = y[
            start:
            start + SEGMENT_SAMPLES
        ]

        segment = pad_or_trim(
            segment
        )

        segments.append(
            segment
        )

        start += SEGMENT_SAMPLES

    return segments


# ============================================================
# Dataset
# ============================================================

class GenreSegmentDataset(Dataset):
    """
    Lazy-loading dataset for genre classification.

    The dataset stores only track metadata and segment indices.
    Audio is loaded only when __getitem__() is called.

    This preserves the original preprocessing pipeline while
    avoiding loading the entire dataset into RAM.
    """

    def __init__(self, rows):
        self.rows = rows
        self.items = []

        print(
            f"Preparing segment index "
            f"for {len(rows)} tracks..."
        )

        for track_index, row in enumerate(rows):

            path = row["absolute_path"]

            # Duration is already available in the frozen manifest.
            duration = float(
                row["duration_seconds"]
            )

            # Preserve the exact behavior of extract_segments():
            # a final partial segment is retained and zero-padded.
            segment_count = max(
                1,
                math.ceil(
                    duration / SEGMENT_SECONDS
                ),
            )

            label = CLASS_TO_INDEX[
                row["genre"]
            ]

            for segment_index in range(
                segment_count
            ):
                self.items.append(
                    (
                        track_index,
                        segment_index,
                        label,
                    )
                )

            if (
                track_index + 1
            ) % 100 == 0:

                print(
                    f"  "
                    f"{track_index + 1}/"
                    f"{len(rows)} tracks"
                )

        print(
            f"Total segments: "
            f"{len(self.items)}"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        (
            track_index,
            segment_index,
            label,
        ) = self.items[index]

        row = self.rows[track_index]

        path = row["absolute_path"]

        # Load ONLY this track when requested.
        y = load_audio(path)

        # Calculate the exact segment boundaries
        # used by the original extract_segments().
        start = (
            segment_index *
            SEGMENT_SAMPLES
        )

        end = (
            start +
            SEGMENT_SAMPLES
        )

        audio_segment = y[start:end]

        # Preserve the original final-segment
        # zero-padding behavior.
        audio_segment = pad_or_trim(
            audio_segment
        )

        mel = make_mel_spectrogram(
            audio_segment
        )

        return (
            mel,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
            track_index,
            segment_index,
        )


# ============================================================
# Model
# ============================================================

class CustomCNN(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.classifier = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1,
        )

        x = self.dropout(x)

        return self.classifier(x)


# ============================================================
# Model smoke test
# ============================================================

def smoke_test_model(
    model,
    device,
):

    print()
    print("=" * 72)
    print("MODEL SMOKE TEST")
    print("=" * 72)

    x = torch.randn(
        4,
        1,
        N_MELS,
        130,
        device=device,
    )

    output = model(x)

    expected = (
        4,
        NUM_CLASSES,
    )

    print(
        f"Input shape  : {tuple(x.shape)}"
    )

    print(
        f"Output shape : {tuple(output.shape)}"
    )

    print(
        f"Expected     : {expected}"
    )

    if tuple(output.shape) != expected:
        raise RuntimeError(
            "Model output shape is incorrect."
        )

    if not torch.isfinite(
        output
    ).all():

        raise RuntimeError(
            "Model produced NaN/Inf."
        )

    print(
        "MODEL SMOKE TEST: PASS"
    )


# ============================================================
# Class weights
# ============================================================

def calculate_class_weights(
    rows,
):

    counts = Counter(
        row["genre"]
        for row in rows
    )

    values = np.array(
        [
            counts.get(
                genre,
                0,
            )
            for genre in CLASS_NAMES
        ],
        dtype=np.float64,
    )

    if np.any(values <= 0):
        missing = [
            CLASS_NAMES[i]
            for i, value in enumerate(
                values
            )
            if value <= 0
        ]

        raise RuntimeError(
            "Training set has no samples for: "
            + ", ".join(missing)
        )

    # Balanced inverse-frequency weights.
    weights = (
        len(rows)
        / (
            NUM_CLASSES * values
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for batch in loader:

        (
            inputs,
            targets,
            _,
            _,
        ) = batch

        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(inputs)

        loss = criterion(
            logits,
            targets,
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite training loss."
            )

        loss.backward()

        optimizer.step()

        batch_size = (
            targets.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += int(
            (
                predictions
                == targets
            ).sum().item()
        )

        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate_segment_level(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    total = 0

    correct = 0

    for batch in loader:

        (
            inputs,
            targets,
            _,
            _,
        ) = batch

        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )

        logits = model(inputs)

        loss = criterion(
            logits,
            targets,
        )

        batch_size = (
            targets.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += int(
            (
                predictions
                == targets
            ).sum().item()
        )

        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print(
        "GENGROGRAM - EXPANDED 20-CLASS CNN"
    )
    print("=" * 72)

    print()
    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print()
    print("Loading manifests...")

    train_rows = load_manifest(
        TRAIN_MANIFEST
    )

    val_rows = load_manifest(
        VAL_MANIFEST
    )

    test_rows = load_manifest(
        TEST_MANIFEST
    )

    verify_manifest(
        train_rows,
        "Train",
    )

    verify_manifest(
        val_rows,
        "Validation",
    )

    verify_manifest(
        test_rows,
        "Test",
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print()
    print("Training class distribution:")

    train_counts = Counter(
        row["genre"]
        for row in train_rows
    )

    for genre in CLASS_NAMES:
        print(
            f"  {genre:20s}: "
            f"{train_counts[genre]}"
        )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_rows
        )
    )

    print()
    print("Class weights:")

    for genre, weight in zip(
        CLASS_NAMES,
        class_weights.tolist(),
    ):

        print(
            f"  {genre:20s}: "
            f"{weight:.4f}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CustomCNN(
        NUM_CLASSES
    ).to(device)

    smoke_test_model(
        model,
        device,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("BUILDING TRAINING DATASET")
    print("=" * 72)

    train_dataset = GenreSegmentDataset(
        train_rows
    )

    print()
    print("=" * 72)
    print("BUILDING VALIDATION DATASET")
    print("=" * 72)

    val_dataset = GenreSegmentDataset(
        val_rows
    )

    # Test dataset is deliberately NOT constructed here.
    # It remains untouched until final evaluation.

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = math.inf

    best_epoch = 0

    patience_counter = 0

    history = []

    start_time = time.time()

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_acc = (
            evaluate_segment_level(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        scheduler.step(
            val_loss
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_seconds": elapsed,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val acc={val_acc:.4f} | "
            f"lr={current_lr:.6f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        CLASS_NAMES,

                    "class_to_index":
                        CLASS_TO_INDEX,

                    "config": {
                        "sample_rate":
                            SAMPLE_RATE,
                        "segment_seconds":
                            SEGMENT_SECONDS,
                        "n_mels":
                            N_MELS,
                        "n_fft":
                            N_FFT,
                        "hop_length":
                            HOP_LENGTH,
                        "num_classes":
                            NUM_CLASSES,
                    },

                    "epoch": epoch,

                    "validation_loss":
                        val_loss,

                    "validation_accuracy":
                        val_acc,
                },
                MODEL_ROOT
                / "best_model.pt",
            )

            print(
                "  -> saved best model"
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    total_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Save history/config
    # --------------------------------------------------------

    with (
        MODEL_ROOT
        / "training_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    training_config = {
        "seed": SEED,
        "sample_rate": SAMPLE_RATE,
        "segment_seconds":
            SEGMENT_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "epochs_requested":
            NUM_EPOCHS,
        "patience": PATIENCE,
        "dropout": DROPOUT,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "train_tracks": len(
            train_rows
        ),
        "validation_tracks": len(
            val_rows
        ),
        "test_tracks": len(
            test_rows
        ),
        "best_epoch": best_epoch,
        "best_validation_loss":
            best_val_loss,
        "training_seconds":
            total_time,
    }

    with (
        MODEL_ROOT
        / "training_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            training_config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best val loss    : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Training time    : "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        "Best model:"
    )

    print(
        MODEL_ROOT
        / "best_model.pt"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The test set has not been evaluated."
    )

    print(
        "Run the dedicated final evaluation "
        "after training."
    )


if __name__ == "__main__":
    main()