import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ml.audio_processor import AudioProcessor

GENRE_TO_IDX = {
    "blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4,
    "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9
}

IDX_TO_GENRE = {v: k for k, v in GENRE_TO_IDX.items()}

class GTZANSegmentDataset(Dataset):
    """
    Segment-level PyTorch Dataset for GTZAN tracks.
    Each 30s track yields multiple 3.0s segments (e.g. 10 segments per track).
    Tracks are strictly filtered by the split manifest to prevent data leakage.
    """
    def __init__(self, manifest_path: str = "data/split_manifest.json", data_dir: str = "data/gtzan", split: str = "train", config_path: str = "configs/default_config.json"):
        super(GTZANSegmentDataset, self).__init__()
        
        self.data_dir = Path(data_dir)
        self.processor = AudioProcessor(config_path)
        self.split = split
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        rel_files = manifest["splits"].get(split, [])
        self.items = [] # list of dicts: {"segment_tensor": tensor, "label": int, "track_id": str}
        
        print(f"Loading {split} dataset split ({len(rel_files)} tracks)...")
        for rel_fp in rel_files:
            full_fp = self.data_dir / rel_fp
            if not full_fp.exists():
                continue
                
            genre = full_fp.parent.name
            if genre not in GENRE_TO_IDX:
                continue
                
            label = GENRE_TO_IDX[genre]
            
            try:
                # tensor shape: [N_segments, 1, 128, 130]
                track_tensor = self.processor.process_file_to_tensor(str(full_fp))
                for seg_idx in range(track_tensor.shape[0]):
                    seg_tensor = track_tensor[seg_idx] # [1, 128, 130]
                    self.items.append({
                        "spectrogram": seg_tensor,
                        "label": label,
                        "genre": genre,
                        "track_id": full_fp.name,
                        "segment_idx": seg_idx
                    })
            except Exception as e:
                print(f"Warning: Failed to process {full_fp}: {e}")

        print(f"Dataset '{split}' loaded: {len(self.items)} 3.0s spectrogram segments from {len(rel_files)} tracks.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item["spectrogram"], item["label"]

def get_dataloader(manifest_path: str, data_dir: str, split: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = GTZANSegmentDataset(manifest_path, data_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

if __name__ == "__main__":
    ds = GTZANSegmentDataset(split="val")
    if len(ds) > 0:
        spec, lbl = ds[0]
        print("Sample Spectrogram Shape:", spec.shape)
        print("Sample Label            :", lbl, f"({IDX_TO_GENRE[lbl]})")
