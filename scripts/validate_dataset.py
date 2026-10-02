import os
import json
import csv
from pathlib import Path
import soundfile as sf

DATA_DIR = Path("data/gtzan")
OUTPUT_DIR = Path("audit_outputs/dataset")
EXPECTED_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

def find_genre_dir(root: Path):
    candidates = list(root.glob("genres*")) + [root]
    for cand in candidates:
        if cand.is_dir():
            subdirs = [d.name for d in cand.iterdir() if d.is_dir()]
            if any(g in subdirs for g in EXPECTED_GENRES):
                return cand
    return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    genre_base = find_genre_dir(DATA_DIR)
    
    if not genre_base:
        print(f"Error: Could not locate genre directories in {DATA_DIR}")
        return

    print(f"Auditing dataset at {genre_base}...")
    
    inventory = []
    genre_counts = {g: 0 for g in EXPECTED_GENRES}
    valid_tracks = 0
    corrupt_tracks = 0
    total_duration = 0.0
    duplicate_filenames = set()
    seen_filenames = set()
    ignored_dot_files = 0

    for genre in EXPECTED_GENRES:
        genre_path = genre_base / genre
        if not genre_path.exists():
            print(f"Warning: Genre directory {genre} missing!")
            continue
            
        audio_files = sorted(list(genre_path.glob("*.wav")) + list(genre_path.glob("*.au")))
        for file_path in audio_files:
            fname = file_path.name
            
            # Skip macOS hidden resource fork files starting with ._
            if fname.startswith("._"):
                ignored_dot_files += 1
                continue

            if fname in seen_filenames:
                duplicate_filenames.add(fname)
            seen_filenames.add(fname)

            file_size = file_path.stat().st_size
            status = "VALID"
            duration = 0.0
            sr = 0
            channels = 0

            try:
                info = sf.info(str(file_path))
                sr = info.samplerate
                channels = info.channels
                duration = info.duration
                if duration < 1.0 or sr == 0:
                    status = "CORRUPT_SHORT"
                    corrupt_tracks += 1
                else:
                    valid_tracks += 1
                    genre_counts[genre] += 1
                    total_duration += duration
            except Exception as e:
                status = f"CORRUPT_{type(e).__name__}"
                corrupt_tracks += 1

            inventory.append({
                "filepath": str(file_path.relative_to(DATA_DIR)),
                "genre": genre,
                "duration": round(duration, 3),
                "sample_rate": sr,
                "channels": channels,
                "file_size": file_size,
                "validation_status": status
            })

    # Save CSV Inventory
    csv_path = Path("gtzan_inventory.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath", "genre", "duration", "sample_rate", "channels", "file_size", "validation_status"
        ])
        writer.writeheader()
        writer.writerows(inventory)

    # Save JSON Report
    report = {
        "dataset_name": "GTZAN",
        "genre_base_path": str(genre_base),
        "total_files_scanned": len(inventory),
        "ignored_mac_resource_fork_files": ignored_dot_files,
        "valid_tracks": valid_tracks,
        "corrupt_tracks": corrupt_tracks,
        "corrupt_track_details": [r["filepath"] for r in inventory if r["validation_status"] != "VALID"],
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_hours": round(total_duration / 3600, 2),
        "duplicate_filenames_count": len(duplicate_filenames),
        "genre_counts": genre_counts,
        "is_class_balanced": len(set(genre_counts.values())) == 1
    }

    report_path = OUTPUT_DIR / "gtzan_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- GTZAN DATASET AUDIT SUMMARY ---")
    print(f"Total Original Audio Tracks: {len(inventory)}")
    print(f"Ignored Mac ._ Metadata Files: {ignored_dot_files}")
    print(f"Valid Audio Tracks: {valid_tracks}")
    print(f"Corrupt Audio Tracks: {corrupt_tracks}")
    print(f"Corrupt Files List: {report['corrupt_track_details']}")
    print(f"Genre Breakdown (Valid): {genre_counts}")
    print(f"Class Balanced: {report['is_class_balanced']}")
    print(f"Report saved to: {report_path}")
    print(f"Inventory saved to: {csv_path}")

if __name__ == "__main__":
    main()
