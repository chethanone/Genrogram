import sys
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ml.audio_processor import AudioProcessor

MANIFEST_PATH = project_root / "data/split_manifest.json"
OUTPUT_DIR = project_root / "audit_outputs/dataset"

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} not found. Please run scripts/create_split_manifest.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    train_files = manifest["splits"]["train"]
    if not train_files:
        print("Error: Train split is empty.")
        return

    # Select exactly ONE valid file from train split
    test_file_rel = train_files[0]
    test_file_path = project_root / "data/gtzan" / test_file_rel

    if not test_file_path.exists():
        print(f"Error: Audio file {test_file_path} does not exist.")
        return

    print(f"Running single-file pipeline test on: {test_file_path}")

    processor = AudioProcessor(str(project_root / "configs/default_config.json"))
    
    # 1. Raw Audio Load & Resample
    y = processor.load_and_preprocess_raw_audio(str(test_file_path))
    print(f"Loaded audio length: {len(y)} samples ({len(y)/processor.sample_rate:.2f}s) @ {processor.sample_rate}Hz")

    # 2. Segmentation
    segments = processor.extract_segments(y)
    print(f"Extracted {len(segments)} segment(s) of duration {processor.segment_duration}s")

    # 3. Process to PyTorch Tensor
    tensor = processor.process_file_to_tensor(str(test_file_path))
    print(f"Output Tensor Shape: {tensor.shape} (N, C, Mel_Bins, Time_Frames)")
    print(f"Tensor Dtype: {tensor.dtype}")
    print(f"Tensor Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}")

    # 4. Check finite values
    is_finite = torch.isfinite(tensor).all().item()
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    print(f"Finite Check: {is_finite} | Has NaN: {has_nan} | Has Inf: {has_inf}")

    assert is_finite and not has_nan and not has_inf, "Tensor contains invalid non-finite numbers!"

    # 5. Plot Diagnostic Spectrogram for 1st segment
    mel_spec = tensor[0, 0].numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-Spectrogram Test — {test_file_path.name} (Segment 1)")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.tight_layout()

    out_plot = OUTPUT_DIR / "single_file_spectrogram_test.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"\nSUCCESS: Diagnostic spectrogram plot saved to {out_plot}")
    print("Single-file pipeline verification PASSED!")

if __name__ == "__main__":
    main()
