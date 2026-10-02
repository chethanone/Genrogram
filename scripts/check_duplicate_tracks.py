import os
import json
import hashlib
from pathlib import Path

MANIFEST_PATH = Path("data/split_manifest.json")
DATA_DIR = Path("data/gtzan")
OUTPUT_PATH = Path("audit_outputs/dataset/duplicate_check_report.json")

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} missing.")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    track_to_split = {}
    for split_name, file_list in splits.items():
        for rel_path in file_list:
            track_to_split[rel_path] = split_name

    print("Computing SHA-256 checksums for all audio tracks in manifest...")
    
    hash_map = {}  # sha256 -> list of rel_paths
    for rel_path in track_to_split.keys():
        full_path = DATA_DIR / rel_path
        if full_path.exists():
            h = compute_sha256(full_path)
            hash_map.setdefault(h, []).append(rel_path)

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    cross_split_leakage = []
    same_split_duplicates = []

    for h, paths in duplicates.items():
        split_names = set(track_to_split[p] for p in paths)
        if len(split_names) > 1:
            cross_split_leakage.append({"hash": h, "tracks": paths, "splits": list(split_names)})
        else:
            same_split_duplicates.append({"hash": h, "tracks": paths, "split": list(split_names)[0]})

    report = {
        "total_manifest_tracks": len(track_to_split),
        "unique_sha256_hashes": len(hash_map),
        "duplicate_hash_clusters": len(duplicates),
        "cross_split_leakage_count": len(cross_split_leakage),
        "same_split_duplicate_count": len(same_split_duplicates),
        "cross_split_leakage_details": cross_split_leakage,
        "same_split_duplicate_details": same_split_duplicates,
        "has_cross_split_leakage": len(cross_split_leakage) > 0
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- STEP 0: SHA-256 DATASET INTEGRITY SUMMARY ---")
    print(f"Total Tracks Evaluated: {report['total_manifest_tracks']}")
    print(f"Unique SHA-256 Hashes: {report['unique_sha256_hashes']}")
    print(f"Exact Binary Duplicates Found: {report['duplicate_hash_clusters']}")
    print(f"Cross-Split Leakage Clusters: {report['cross_split_leakage_count']}")
    print(f"Cross-Split Leakage Detected: {report['has_cross_split_leakage']}")
    print(f"Detailed Report saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
