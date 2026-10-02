import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import time
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from ml.custom_cnn import CustomCNN
from ml.dataset import GTZANSegmentDataset

MANIFEST_PATH = project_root / "data/split_manifest.json"
DATA_DIR = project_root / "data/gtzan"
CONFIG_PATH = project_root / "configs/default_config.json"
BENCHMARK_CKPT = project_root / "models/benchmark_checkpoint.pt"

def main():
    print("==================================================")
    print("      STARTING CONTROLLED 1-EPOCH BENCHMARK       ")
    print("==================================================")

    # 1. Device check
    device = torch.device("cpu")
    process = psutil.Process()
    cpu_percent_start = psutil.cpu_percent(interval=None)

    # 2. Datasets & Loaders
    batch_size = 32
    print(f"Loading full Training DataLoader (698 tracks, batch size {batch_size})...")
    train_dataset = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="train", config_path=str(CONFIG_PATH))
    print(f"Loading full Validation DataLoader (144 tracks, batch size {batch_size})...")
    val_dataset = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="val", config_path=str(CONFIG_PATH))

    num_workers = 0 if sys.platform.startswith("win") else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = CustomCNN(num_classes=10, in_channels=1, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_train_batches = len(train_loader)
    print(f"\nTotal Train Segments: {len(train_dataset)} | Batches: {num_train_batches}")
    print(f"Total Val Segments  : {len(val_dataset)} | Batches: {len(val_loader)}")

    # 3. Train Phase Timing
    train_start_time = time.time()
    model.train()
    running_train_loss = 0.0
    correct_train = 0

    batch_times = []

    for b_idx, (b_inputs, b_labels) in enumerate(train_loader):
        b_start = time.time()
        b_inputs, b_labels = b_inputs.to(device), b_labels.to(device)

        optimizer.zero_grad()
        outputs = model(b_inputs)
        loss = criterion(outputs, b_labels)
        loss.backward()
        optimizer.step()

        b_dur = time.time() - b_start
        batch_times.append(b_dur)

        running_train_loss += loss.item() * b_inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == b_labels).sum().item()

        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == num_train_batches:
            avg_b_time = sum(batch_times[-50:]) / len(batch_times[-50:])
            print(f" Batch [{b_idx+1:03d}/{num_train_batches:03d}] | Avg Batch Time: {avg_b_time:.3f}s | Current Batch Loss: {loss.item():.4f}")

    train_duration = time.time() - train_start_time
    epoch_train_loss = running_train_loss / len(train_dataset)
    epoch_train_acc = correct_train / len(train_dataset)

    # 4. Validation Phase Timing
    val_start_time = time.time()
    model.eval()
    running_val_loss = 0.0
    correct_val = 0

    with torch.no_grad():
        for b_inputs, b_labels in val_loader:
            b_inputs, b_labels = b_inputs.to(device), b_labels.to(device)
            outputs = model(b_inputs)
            loss = criterion(outputs, b_labels)

            running_val_loss += loss.item() * b_inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == b_labels).sum().item()

    val_duration = time.time() - val_start_time
    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_acc = correct_val / len(val_dataset)

    total_epoch_time = train_duration + val_duration
    avg_batch_time = sum(batch_times) / len(batch_times)

    # Resource metrics
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)
    cpu_percent_end = psutil.cpu_percent(interval=1.0)

    # Save temporary benchmark checkpoint
    BENCHMARK_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": 1,
        "is_benchmark_checkpoint": True,
        "model_state_dict": model.state_dict(),
        "train_loss": epoch_train_loss,
        "val_loss": epoch_val_loss
    }, BENCHMARK_CKPT)

    # Projections
    time_10_epochs_sec = total_epoch_time * 10
    time_20_epochs_sec = total_epoch_time * 20
    time_30_epochs_sec = total_epoch_time * 30

    report = {
        "device": "cpu",
        "total_train_segments": len(train_dataset),
        "total_val_segments": len(val_dataset),
        "batch_size": batch_size,
        "total_epoch_runtime_sec": round(total_epoch_time, 2),
        "train_runtime_sec": round(train_duration, 2),
        "val_runtime_sec": round(val_duration, 2),
        "avg_batch_runtime_sec": round(avg_batch_time, 4),
        "ram_usage_mb": round(ram_mb, 2),
        "cpu_usage_percent": cpu_percent_end,
        "train_loss": round(epoch_train_loss, 4),
        "train_accuracy": round(epoch_train_acc, 4),
        "val_loss": round(epoch_val_loss, 4),
        "val_accuracy": round(epoch_val_acc, 4),
        "estimated_runtimes": {
            "10_epochs_min": round(time_10_epochs_sec / 60, 2),
            "20_epochs_min": round(time_20_epochs_sec / 60, 2),
            "30_epochs_min": round(time_30_epochs_sec / 60, 2)
        }
    }

    out_json = project_root / "audit_outputs/dataset/benchmark_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n==================================================")
    print("         BENCHMARK RESULTS & PROJECTIONS          ")
    print("==================================================")
    print(f" Train Runtime      : {train_duration:.2f}s")
    print(f" Val Runtime        : {val_duration:.2f}s")
    print(f" Total Epoch Time   : {total_epoch_time:.2f}s ({total_epoch_time/60:.2f} min)")
    print(f" Avg Batch Time     : {avg_batch_time:.4f}s")
    print(f" RAM Usage          : {ram_mb:.1f} MB")
    print(f" CPU Usage          : {cpu_percent_end}%")
    print(f" Train Loss         : {epoch_train_loss:.4f}")
    print(f" Train Accuracy     : {epoch_train_acc*100:.2f}%")
    print(f" Val Loss           : {epoch_val_loss:.4f}")
    print(f" Val Accuracy       : {epoch_val_acc*100:.2f}%")
    print("--------------------------------------------------")
    print(f" Estimated 10 Epochs: {time_10_epochs_sec/60:.2f} min")
    print(f" Estimated 20 Epochs: {time_20_epochs_sec/60:.2f} min")
    print(f" Estimated 30 Epochs: {time_30_epochs_sec/60:.2f} min ({time_30_epochs_sec/3600:.2f} hours)")
    print("==================================================")

if __name__ == "__main__":
    main()
