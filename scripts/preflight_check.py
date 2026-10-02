import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import torch
from ml.audio_processor import AudioProcessor
from ml.dataset import GTZANSegmentDataset

MANIFEST_PATH = project_root / "data/split_manifest.json"
DATA_DIR = project_root / "data/gtzan"
CONFIG_PATH = project_root / "configs/default_config.json"

def main():
    print("Executing Preflight Data & Device Audit...")
    
    # 1. Device check
    cuda_available = torch.cuda.is_available()
    selected_device = "cuda" if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Intel / Host)"
    mixed_precision = cuda_available

    # 2. Config settings
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    batch_size = config.get("model", {}).get("batch_size", 32)
    epochs = config.get("model", {}).get("epochs", 30)

    # 3. Track & Segment counts
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    train_tracks = len(splits["train"])
    val_tracks = len(splits["val"])
    test_tracks = len(splits["test"])

    train_ds = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="train")
    val_ds = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="val")
    test_ds = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="test")

    train_segs = len(train_ds)
    val_segs = len(val_ds)
    test_segs = len(test_ds)

    est_batches_per_epoch = (train_segs + batch_size - 1) // batch_size
    est_checkpoint_size_mb = 1.6 # ~400k float32 params + optimizer state
    est_disk_usage_mb = 10.0

    report = {
        "cuda_available": cuda_available,
        "selected_device": selected_device,
        "device_name": device_name,
        "mixed_precision": mixed_precision,
        "counts": {
            "train_tracks": train_tracks,
            "val_tracks": val_tracks,
            "test_tracks": test_tracks,
            "train_segments": train_segs,
            "val_segments": val_segs,
            "test_segments": test_segs,
            "total_tracks": train_tracks + val_tracks + test_tracks,
            "total_segments": train_segs + val_segs + test_segs
        },
        "training_params": {
            "batch_size": batch_size,
            "max_epochs": epochs,
            "batches_per_epoch": est_batches_per_epoch,
            "expected_checkpoint_location": "models/best_model.pt",
            "estimated_disk_usage_mb": est_disk_usage_mb
        }
    }

    out_file = project_root / "audit_outputs/dataset/preflight_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n==================================================")
    print("             PREFLIGHT AUDIT REPORT               ")
    print("==================================================")
    print(f" 1. CUDA Availability       : {cuda_available}")
    print(f" 2. Selected Device         : {selected_device} ({device_name})")
    print(f" 3. Train Track Count       : {train_tracks} tracks")
    print(f" 4. Validation Track Count  : {val_tracks} tracks")
    print(f" 5. Test Track Count        : {test_tracks} tracks")
    print(f" 6. Train Segment Count     : {train_segs} 3s segments")
    print(f" 7. Validation Segment Count: {val_segs} 3s segments")
    print(f" 8. Test Segment Count      : {test_segs} 3s segments")
    print(f" 9. Batch Size              : {batch_size}")
    print(f"10. Batches per Epoch       : {est_batches_per_epoch}")
    print(f"11. Checkpoint Path         : models/best_model.pt")
    print(f"12. Estimated Disk Usage    : ~{est_disk_usage_mb} MB")
    print(f"13. Mixed Precision (AMP)   : {mixed_precision}")
    print("==================================================")

if __name__ == "__main__":
    main()
