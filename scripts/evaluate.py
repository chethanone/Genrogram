import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

from ml.custom_cnn import CustomCNN
from ml.audio_processor import AudioProcessor
from ml.dataset import GENRE_TO_IDX, IDX_TO_GENRE

MANIFEST_PATH = project_root / "data/split_manifest.json"
DATA_DIR = project_root / "data/gtzan"
MODEL_PATH = project_root / "models/best_model.pt"
OUTPUT_JSON = project_root / "models/evaluation_results.json"
RESULTS_DIR = project_root / "results"

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Trained model checkpoint {MODEL_PATH} not found. Please train model first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Model Checkpoint
    print(f"Loading best model checkpoint from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = CustomCNN(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    processor = AudioProcessor(str(project_root / "configs/default_config.json"))

    # 2. Load Test Split Manifest
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    test_files = manifest["splits"].get("test", [])
    print(f"Evaluating untouched TEST set ({len(test_files)} original tracks)...")

    track_true_labels = []
    track_pred_labels = []

    seg_true_labels = []
    seg_pred_labels = []

    # 3. Predict on Test Tracks with Segment Aggregation (Mean Probabilities)
    with torch.no_grad():
        for rel_fp in test_files:
            full_fp = DATA_DIR / rel_fp
            if not full_fp.exists():
                continue

            genre_name = full_fp.parent.name
            if genre_name not in GENRE_TO_IDX:
                continue

            true_label = GENRE_TO_IDX[genre_name]

            # Process audio to segment tensors: [N_segments, 1, 128, 130]
            track_tensor = processor.process_file_to_tensor(str(full_fp))

            # Get segment probabilities via Softmax
            logits = model(track_tensor)
            probs = torch.softmax(logits, dim=-1).numpy() # [N_segments, 10]

            # Track-Level Aggregation: Mean probability vector across 3.0s segments
            mean_track_prob = np.mean(probs, axis=0)
            track_pred_label = int(np.argmax(mean_track_prob))

            track_true_labels.append(true_label)
            track_pred_labels.append(track_pred_label)

            # Segment-level logging
            for p_seg in probs:
                seg_true_labels.append(true_label)
                seg_pred_labels.append(int(np.argmax(p_seg)))

    # 4. Calculate Original Track-Level Evaluation Metrics
    track_acc = accuracy_score(track_true_labels, track_pred_labels)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        track_true_labels, track_pred_labels, average='macro', zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        track_true_labels, track_pred_labels, labels=list(range(10)), zero_division=0
    )

    cm_track = confusion_matrix(track_true_labels, track_pred_labels, labels=list(range(10)))

    # Calculate Segment-Level Metrics
    seg_acc = accuracy_score(seg_true_labels, seg_pred_labels)
    _, _, seg_macro_f1, _ = precision_recall_fscore_support(
        seg_true_labels, seg_pred_labels, average='macro', zero_division=0
    )

    genre_names = [IDX_TO_GENRE[i] for i in range(10)]

    per_class_metrics = {}
    for i, g in enumerate(genre_names):
        per_class_metrics[g] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    results = {
        "evaluation_scope": "original_track_level_untouched_test_set",
        "total_test_tracks": len(track_true_labels),
        "total_test_segments": len(seg_true_labels),
        "track_level_metrics": {
            "accuracy": round(float(track_acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4)
        },
        "segment_level_metrics": {
            "accuracy": round(float(seg_acc), 4),
            "macro_f1": round(float(seg_macro_f1), 4)
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm_track.tolist()
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Render Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_track, annot=True, fmt='d', cmap='Reds',
        xticklabels=genre_names, yticklabels=genre_names, cbar=False
    )
    plt.title("GENROGRAM — Track-Level Confusion Matrix (Test Set)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Genre", fontsize=11)
    plt.ylabel("True Genre", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Render Per-Class F1 Bar Chart
    plt.figure(figsize=(10, 5))
    f1_values = [per_class_f1[i] * 100 for i in range(10)]
    plt.bar(genre_names, f1_values, color="#C96F70")
    plt.title("Per-Class F1-Score (%) — Test Set Evaluation", fontsize=12, fontweight='bold')
    plt.xlabel("Genre")
    plt.ylabel("F1-Score (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_f1.png", dpi=150)
    plt.close()

    print("\n==================================================")
    print("      GENROGRAM TEST SET EVALUATION REPORT        ")
    print("==================================================")
    print(f"Total Test Tracks Evaluated: {len(track_true_labels)}")
    print(f"Track-Level Accuracy      : {track_acc*100:.2f}%")
    print(f"Track-Level Macro F1-Score: {macro_f1:.4f}")
    print(f"Track-Level Macro Precision: {macro_p:.4f}")
    print(f"Track-Level Macro Recall   : {macro_r:.4f}")
    print("--------------------------------------------------")
    print(f"Segment-Level Accuracy    : {seg_acc*100:.2f}%")
    print(f"Segment-Level Macro F1    : {seg_macro_f1:.4f}")
    print("==================================================")
    print(f"Evaluation JSON saved to: {OUTPUT_JSON}")
    print(f"Confusion Matrix saved to: {RESULTS_DIR / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
