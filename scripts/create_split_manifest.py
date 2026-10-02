import os
import json
import csv
import random
import hashlib
from pathlib import Path

INPUT_CSV = Path("gtzan_inventory.csv")
DATA_DIR = Path("data/gtzan")
OUTPUT_JSON = Path("data/split_manifest.json")
OUTPUT_CSV = Path("data/split_manifest.csv")
SEED = 42

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Please run scripts/validate_dataset.py first.")
        return

    # 1. Read valid tracks
    valid_tracks = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["validation_status"] == "VALID":
                valid_tracks.append(row["filepath"])

    # 2. Group by SHA-256 hash to find binary duplicate clusters
    hash_clusters = {}  # sha256 -> list of filepaths
    for rel_path in valid_tracks:
        full_path = DATA_DIR / rel_path
        h = compute_sha256(full_path)
        hash_clusters.setdefault(h, []).append(rel_path)

    # 3. Create unique items for splitting (each item represents a cluster of identical audio)
    # Group clusters by primary genre
    genre_clusters = {}
    for h, files in hash_clusters.items():
        primary_file = files[0]
        genre = Path(primary_file).parent.name
        genre_clusters.setdefault(genre, []).append((h, files))

    train_files, val_files, test_files = [], [], []
    split_details = {}

    random.seed(SEED)

    for genre, clusters in sorted(genre_clusters.items()):
        sorted_clusters = sorted(clusters, key=lambda x: x[0])  # Sort by hash for determinism
        random.shuffle(sorted_clusters)

        n_clusters = len(sorted_clusters)
        n_train = int(n_clusters * 0.70)
        n_val = int(n_clusters * 0.15)

        tr_c = sorted_clusters[:n_train]
        val_c = sorted_clusters[n_train:n_train + n_val]
        te_c = sorted_clusters[n_train + n_val:]

        # Expand cluster files into their respective splits
        g_train = [f for _, flist in tr_c for f in flist]
        g_val = [f for _, flist in val_c for f in flist]
        g_test = [f for _, flist in te_c for f in flist]

        train_files.extend(g_train)
        val_files.extend(g_val)
        test_files.extend(g_test)

        split_details[genre] = {
            "total_tracks": len(g_train) + len(g_val) + len(g_test),
            "train": len(g_train),
            "val": len(g_val),
            "test": len(g_test)
        }

    # 4. Rigorous verification: Ensure ZERO SHA-256 hash overlap between splits
    train_hashes = set(compute_sha256(DATA_DIR / f) for f in train_files)
    val_hashes = set(compute_sha256(DATA_DIR / f) for f in val_files)
    test_hashes = set(compute_sha256(DATA_DIR / f) for f in test_files)

    intersect_tr_val = train_hashes.intersection(val_hashes)
    intersect_tr_te = train_hashes.intersection(test_hashes)
    intersect_val_te = val_hashes.intersection(test_hashes)

    has_leakage = len(intersect_tr_val) > 0 or len(intersect_tr_te) > 0 or len(intersect_val_te) > 0

    manifest = {
        "split_policy": "original_track_and_cluster_aware",
        "random_seed": SEED,
        "ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_tracks": len(train_files) + len(val_files) + len(test_files),
        "counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        },
        "leakage_verification": {
            "has_leakage": has_leakage,
            "train_val_hash_intersection": len(intersect_tr_val),
            "train_test_hash_intersection": len(intersect_tr_te),
            "val_test_hash_intersection": len(intersect_val_te)
        },
        "per_genre_counts": split_details,
        "splits": {
            "train": sorted(train_files),
            "val": sorted(val_files),
            "test": sorted(test_files)
        }
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Export CSV version
    csv_rows = []
    for split_name, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for fp in flist:
            genre = Path(fp).parent.name
            csv_rows.append({"filepath": fp, "genre": genre, "split": split_name})

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "genre", "split"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n--- CLUSTER-AWARE TRACK SPLIT MANIFEST SUMMARY ---")
    print(f"Seed: {SEED}")
    print(f"Train tracks: {len(train_files)}")
    print(f"Val tracks:   {len(val_files)}")
    print(f"Test tracks:  {len(test_files)}")
    print(f"Binary Hash Leakage Check Passed: {not has_leakage}")
    print(f"Manifest JSON saved to: {OUTPUT_JSON}")
    print(f"Manifest CSV saved to:  {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
