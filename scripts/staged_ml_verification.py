import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from ml.custom_cnn import CustomCNN
from ml.dataset import GTZANSegmentDataset, GENRE_TO_IDX, IDX_TO_GENRE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MANIFEST_PATH = project_root / "data/split_manifest.json"
DATA_DIR = project_root / "data/gtzan"
MODEL_DIR = project_root / "models"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"Executing Staged ML Verification on Device: {device}")

    # =========================================================================
    # STEP 3 — SINGLE BATCH TEST
    # =========================================================================
    print("\n--- STEP 3: SINGLE BATCH TEST ---")
    val_dataset = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    inputs, labels = next(iter(val_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    print(f"Input Tensor Shape: {inputs.shape} (Expected: [batch, 1, 128, 130])")
    print(f"Input Tensor Dtype: {inputs.dtype}")
    print(f"Input Tensor Range: min={inputs.min():.4f}, max={inputs.max():.4f}")
    print(f"Labels Tensor Shape: {labels.shape}")
    print(f"Labels Value Range : min={labels.min().item()}, max={labels.max().item()} (0 to 9)")
    
    assert inputs.shape[1:] == torch.Size([1, 128, 130]), f"Unexpected input shape {inputs.shape}"
    assert torch.isfinite(inputs).all(), "Inputs contain non-finite numbers!"
    assert labels.min() >= 0 and labels.max() < 10, "Labels out of range 0-9!"
    print("STEP 3 PASSED!")

    # =========================================================================
    # STEP 4 — SINGLE FORWARD PASS
    # =========================================================================
    print("\n--- STEP 4: SINGLE FORWARD PASS ---")
    model = CustomCNN(num_classes=10, in_channels=1, dropout=0.3).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(inputs)

    print(f"Logits Tensor Shape: {logits.shape} (Expected: [16, 10])")
    print(f"Logits Finite Check : {torch.isfinite(logits).all().item()}")
    print(f"Logits Has NaN      : {torch.isnan(logits).any().item()}")
    print(f"Logits Has Inf      : {torch.isinf(logits).any().item()}")

    assert logits.shape == torch.Size([16, 10]), f"Unexpected logits shape {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite numbers!"
    print("STEP 4 PASSED!")

    # =========================================================================
    # STEP 5 — SINGLE LOSS CALCULATION & BACKWARD PASS
    # =========================================================================
    print("\n--- STEP 5: SINGLE LOSS CALCULATION ---")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    logits = model(inputs)
    loss = criterion(logits, labels)
    print(f"Calculated Loss Value: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    has_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for name, p in model.named_parameters() if p.requires_grad)
    print(f"Backward Pass Succeeded & Gradients Finite: {has_grads}")
    assert has_grads, "Gradients missing or non-finite!"
    optimizer.zero_grad()
    print("STEP 5 PASSED!")

    # =========================================================================
    # STEP 6 — ONE-BATCH OPTIMIZATION TEST
    # =========================================================================
    print("\n--- STEP 6: ONE-BATCH OPTIMIZATION TEST ---")
    initial_loss = loss.item()
    losses = []
    
    # Run 5 optimization steps on the single batch
    for step in range(5):
        optimizer.zero_grad()
        out = model(inputs)
        l = criterion(out, labels)
        l.backward()
        optimizer.step()
        losses.append(l.item())
        print(f" Optimization Step {step+1}/5 | Loss: {l.item():.4f}")

    print(f"Initial Loss: {initial_loss:.4f} -> Final Batch Loss: {losses[-1]:.4f}")
    assert losses[-1] < initial_loss, "Optimization step did not decrease single-batch loss!"
    print("STEP 6 PASSED!")

    # =========================================================================
    # STEP 7 & 8 & 9 — ONE-EPOCH SMOKE TEST ON A SMALL SUBSET
    # =========================================================================
    print("\n--- STEP 7-9: ONE-EPOCH SMOKE TEST & CHECKPOINTING ---")
    train_dataset = GTZANSegmentDataset(str(MANIFEST_PATH), str(DATA_DIR), split="train")
    
    # Create deterministic small subset of 100 segments for train & 40 for val
    subset_train = Subset(train_dataset, range(0, min(100, len(train_dataset))))
    subset_val = Subset(val_dataset, range(0, min(40, len(val_dataset))))

    smoke_train_loader = DataLoader(subset_train, batch_size=16, shuffle=True)
    smoke_val_loader = DataLoader(subset_val, batch_size=16, shuffle=False)

    model = CustomCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train 1 epoch
    model.train()
    running_loss = 0.0
    for b_inputs, b_labels in smoke_train_loader:
        b_inputs, b_labels = b_inputs.to(device), b_labels.to(device)
        optimizer.zero_grad()
        b_out = model(b_inputs)
        b_loss = criterion(b_out, b_labels)
        b_loss.backward()
        optimizer.step()
        running_loss += b_loss.item() * b_inputs.size(0)

    train_loss = running_loss / len(subset_train)

    # Validate 1 epoch
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for b_inputs, b_labels in smoke_val_loader:
            b_inputs, b_labels = b_inputs.to(device), b_labels.to(device)
            b_out = model(b_inputs)
            b_loss = criterion(b_out, b_labels)
            val_loss += b_loss.item() * b_inputs.size(0)
            preds = torch.argmax(b_out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(b_labels.cpu().numpy())

    val_loss = val_loss / len(subset_val)
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

    print("\n--- SMOKE TEST METRICS (EXPLICITLY LABELED: SMOKE TEST METRICS ONLY) ---")
    print(f"Smoke Test Epoch 1/1 Results:")
    print(f"  Train Loss          : {train_loss:.4f}")
    print(f"  Val Loss            : {val_loss:.4f}")
    print(f"  Val Accuracy        : {acc:.4f}")
    print(f"  Val Macro Precision : {prec:.4f}")
    print(f"  Val Macro Recall    : {rec:.4f}")
    print(f"  Val Macro F1        : {f1:.4f}")

    # Checkpoint saving test (STEP 9)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": acc,
        "val_macro_f1": f1,
        "class_mapping": GENRE_TO_IDX,
        "config": {
            "num_classes": 10,
            "architecture": "CustomCNN",
            "is_smoke_test": True
        }
    }
    
    ckpt_path = MODEL_DIR / "best_model.pt"
    torch.save(checkpoint, ckpt_path)

    history = {
        "smoke_test": True,
        "epochs": [1],
        "train_loss": [round(train_loss, 4)],
        "val_loss": [round(val_loss, 4)],
        "val_accuracy": [round(acc, 4)],
        "val_macro_f1": [round(f1, 4)]
    }

    hist_path = MODEL_DIR / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved Checkpoint to: {ckpt_path}")
    print(f"Saved History to   : {hist_path}")
    print("STEP 7-9 PASSED!")

if __name__ == "__main__":
    main()
