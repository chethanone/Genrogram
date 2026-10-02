# GENROGRAM — Model Training & Safety Verification Protocols

## Staged Verification Workflow

To ensure stability and resource efficiency, training follows a 5-stage verification sequence:

1. **Stage 1:** Single-file audio preprocessing & Mel spectrogram extraction test.
2. **Stage 2:** PyTorch Dataset loader & batch tensor shape verification.
3. **Stage 3:** Model instantiation, forward pass & loss calculation test.
4. **Stage 4:** 1-epoch smoke test on a mini-batch subset.
5. **Stage 5:** Full model training with early stopping & checkpointing.

## Training Configuration

- **Optimizer:** Adam (`lr=1e-3`, `weight_decay=1e-4`)
- **Loss Function:** Cross-Entropy Loss
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping:** Patience = 7 epochs based on validation loss
- **Checkpoints:** Saved to `models/best_model.pt` and `models/final_model.pt`
