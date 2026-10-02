import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from ml.custom_cnn import CustomCNN
from ml.dataset import GTZANSegmentDataset, GENRE_TO_IDX

MANIFEST_PATH = project_root / "data/split_manifest.json"
DATA_DIR = project_root / "data/gtzan"
CONFIG_PATH = project_root / "configs/default_config.json"
MODEL_DIR = project_root / "models"
RESULTS_DIR = project_root / "results"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # 1. Device & CUDA Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Device Selected: {device} | CUDA Available: {torch.cuda.is_available()} | Mixed Precision (AMP): {use_amp}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    # 2. Seed configuration
    seed = config.get("model", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 3. Load Datasets
    batch_size = config.get("model", {}).get("batch_size", 32)
    epochs = config.get("model", {}).get("epochs", 30)
    lr = config.get("model", {}).get("learning_rate", 0.001)
    patience = config.get("model", {}).get("patience", 7)

    train_dataset = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="train", config_path=str(CONFIG_PATH))
    val_dataset = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="val", config_path=str(CONFIG_PATH))

    num_workers = 0 if sys.platform.startswith("win") else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 4. Model, Loss, Optimizer, Scheduler, AMP Scaler
    model = CustomCNN(num_classes=10, in_channels=1, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Save Class Mapping and Training Config
    with open(MODEL_DIR / "class_mapping.json", "w") as f:
        json.dump(GENRE_TO_IDX, f, indent=2)

    training_config = {
        "architecture": "CustomCNN",
        "num_classes": 10,
        "sample_rate": 22050,
        "n_mels": 128,
        "segment_duration": 3.0,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "patience": patience,
        "device": str(device),
        "mixed_precision": use_amp
    }
    with open(MODEL_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # 5. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rates": []
    }

    print(f"\nBeginning Training Routine ({epochs} Max Epochs)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for b_inputs, b_labels in train_loader:
            b_inputs, b_labels = b_inputs.to(device), b_labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(b_inputs)
                    loss = criterion(outputs, b_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(b_inputs)
                loss = criterion(outputs, b_labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * b_inputs.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for b_inputs, b_labels in val_loader:
                b_inputs, b_labels = b_inputs.to(device), b_labels.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(b_inputs)
                        loss = criterion(outputs, b_labels)
                else:
                    outputs = model(b_inputs)
                    loss = criterion(outputs, b_labels)

                val_loss += loss.item() * b_inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(b_labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_acc = accuracy_score(val_targets, val_preds)
        _, _, epoch_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='macro', zero_division=0)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(epoch_val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_accuracy"].append(round(epoch_acc, 4))
        history["val_macro_f1"].append(round(epoch_f1, 4))
        history["learning_rates"].append(current_lr)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_acc*100:.2f}% | Val F1: {epoch_f1:.4f} | LR: {current_lr}")

        # Checkpointing
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_acc,
            "val_macro_f1": epoch_f1,
            "class_mapping": GENRE_TO_IDX,
            "config": training_config
        }

        # Save final checkpoint every epoch
        torch.save(checkpoint_data, MODEL_DIR / "final_model.pt")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(checkpoint_data, MODEL_DIR / "best_model.pt")
            print(f"  --> Saved new best checkpoint to {MODEL_DIR / 'best_model.pt'} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without validation loss improvement.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")

    # Save History JSON
    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 6. Generate Training & Validation Loss/Accuracy Curves
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["epochs"], history["train_loss"], label="Train Loss", color="#161616", linewidth=2)
    plt.plot(history["epochs"], history["val_loss"], label="Val Loss", color="#C96F70", linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save separate loss plot
    plt.savefig(RESULTS_DIR / "training_loss.png", dpi=150)

    # Accuracy Plot
    plt.clf()
    plt.plot(history["epochs"], [acc * 100 for acc in history["val_accuracy"]], label="Val Accuracy (%)", color="#C96F70", linewidth=2)
    plt.title("Validation Accuracy", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(RESULTS_DIR / "validation_accuracy.png", dpi=150)
    plt.close()

    print(f"Saved training history to {MODEL_DIR / 'history.json'}")
    print(f"Saved plots to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
