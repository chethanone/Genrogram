"""
GENGROGRAM - Expanded Audio Dataset Audit

Purpose:
    Audit the actual audio properties of the 10-class expansion dataset.

IMPORTANT:
    - READ-ONLY with respect to raw audio.
    - Does NOT move, rename, modify, convert, or delete audio files.
    - Produces an audit JSON and CSV under audit_outputs/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_ROOT = PROJECT_ROOT / "data" / "expanded_raw"
OUTPUT_ROOT = PROJECT_ROOT / "audit_outputs" / "expansion"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_ROOT / "audio_inventory.csv"
JSON_PATH = OUTPUT_ROOT / "audio_audit_summary.json"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without modifying the source file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_float(value):
    """Convert numpy scalar values safely to Python float."""
    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def get_class_from_path(path: Path) -> str:
    """
    Expected structure:

        expanded_raw/
            Amapiano/
            Bollywood/
            ...

    The first directory below AUDIO_ROOT is treated as the class.
    """
    relative = path.relative_to(AUDIO_ROOT)

    if len(relative.parts) < 2:
        return "UNKNOWN"

    return relative.parts[0]


def rms_db(y: np.ndarray) -> float | None:
    """Calculate RMS level in dBFS-like scale."""
    if y.size == 0:
        return None

    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))

    if rms <= 0:
        return -120.0

    return float(20.0 * np.log10(rms))


def main():
    print("=" * 70)
    print("GENGROGRAM - EXPANDED AUDIO DATASET AUDIT")
    print("=" * 70)
    print(f"Audio root : {AUDIO_ROOT}")
    print(f"Output dir : {OUTPUT_ROOT}")
    print()

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(
            f"Audio root does not exist: {AUDIO_ROOT}"
        )

    files = sorted(
        p
        for p in AUDIO_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    print(f"Audio files discovered: {len(files)}")
    print()

    records = []

    class_counts = Counter()
    extension_counts = Counter()
    sample_rate_counts = Counter()
    channel_counts = Counter()

    invalid_files = []
    short_files = []
    long_files = []
    silent_files = []
    low_energy_files = []

    duration_values = []

    hash_to_files = defaultdict(list)

    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(AUDIO_ROOT)

        genre = get_class_from_path(path)

        record = {
            "relative_path": str(relative_path),
            "genre": genre,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "valid": False,
            "duration_seconds": None,
            "sample_rate": None,
            "channels": None,
            "frames": None,
            "subtype": None,
            "rms_db": None,
            "peak": None,
            "sha256": None,
            "error": None,
        }

        try:
            # Read metadata and audio without modifying the source.
            info = sf.info(str(path))

            record["sample_rate"] = int(info.samplerate)
            record["channels"] = int(info.channels)
            record["frames"] = int(info.frames)
            record["subtype"] = str(info.subtype)

            duration = (
                float(info.frames) / float(info.samplerate)
                if info.samplerate > 0
                else 0.0
            )

            record["duration_seconds"] = safe_float(duration)

            # Load mono audio only for signal-level checks.
            y, sr = librosa.load(
                str(path),
                sr=None,
                mono=True,
            )

            if y.size == 0:
                raise ValueError("Audio contains zero samples")

            peak = float(np.max(np.abs(y)))
            level = rms_db(y)

            record["rms_db"] = safe_float(level)
            record["peak"] = safe_float(peak)
            record["valid"] = True

            class_counts[genre] += 1
            extension_counts[path.suffix.lower()] += 1
            sample_rate_counts[int(sr)] += 1
            channel_counts[int(info.channels)] += 1

            if record["duration_seconds"] is not None:
                duration_values.append(record["duration_seconds"])

                # These are audit flags, not automatic rejection rules.
                if record["duration_seconds"] < 2.0:
                    short_files.append(str(relative_path))

                if record["duration_seconds"] > 600.0:
                    long_files.append(str(relative_path))

            if level is not None:
                if level <= -80.0:
                    silent_files.append(str(relative_path))

                if level <= -45.0:
                    low_energy_files.append(str(relative_path))

            # Exact content duplicate detection.
            file_hash = sha256_file(path)
            record["sha256"] = file_hash
            hash_to_files[file_hash].append(str(relative_path))

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            invalid_files.append(str(relative_path))

        records.append(record)

        if index % 25 == 0 or index == len(files):
            print(
                f"[{index:4d}/{len(files)}] "
                f"processed"
            )

    duplicate_clusters = [
        paths
        for paths in hash_to_files.values()
        if len(paths) > 1
    ]

    valid_records = [
        r for r in records
        if r["valid"]
    ]

    valid_durations = [
        r["duration_seconds"]
        for r in valid_records
        if r["duration_seconds"] is not None
    ]

    if valid_durations:
        duration_array = np.asarray(valid_durations, dtype=np.float64)

        duration_stats = {
            "min_seconds": safe_float(np.min(duration_array)),
            "p25_seconds": safe_float(np.percentile(duration_array, 25)),
            "median_seconds": safe_float(np.median(duration_array)),
            "mean_seconds": safe_float(np.mean(duration_array)),
            "p75_seconds": safe_float(np.percentile(duration_array, 75)),
            "max_seconds": safe_float(np.max(duration_array)),
        }
    else:
        duration_stats = {}

    summary = {
        "total_files": len(records),
        "valid_files": len(valid_records),
        "invalid_files": len(invalid_files),

        "class_counts": dict(sorted(class_counts.items())),

        "extension_counts": dict(sorted(extension_counts.items())),

        "sample_rate_counts": {
            str(k): v
            for k, v in sorted(sample_rate_counts.items())
        },

        "channel_counts": {
            str(k): v
            for k, v in sorted(channel_counts.items())
        },

        "duration_statistics_seconds": duration_stats,

        "short_files_under_2_seconds": {
            "count": len(short_files),
            "files": short_files,
        },

        "long_files_over_10_minutes": {
            "count": len(long_files),
            "files": long_files,
        },

        "silent_files_rms_le_minus_80_db": {
            "count": len(silent_files),
            "files": silent_files,
        },

        "low_energy_files_rms_le_minus_45_db": {
            "count": len(low_energy_files),
            "files": low_energy_files,
        },

        "duplicate_clusters": duplicate_clusters,

        "unique_sha256_hashes": len(hash_to_files),

        "audit_outputs": {
            "inventory_csv": str(CSV_PATH),
            "summary_json": str(JSON_PATH),
        },
    }

    # CSV inventory.
    fieldnames = [
        "relative_path",
        "genre",
        "extension",
        "file_size_bytes",
        "valid",
        "duration_seconds",
        "sample_rate",
        "channels",
        "frames",
        "subtype",
        "rms_db",
        "peak",
        "sha256",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"Total files       : {len(records)}")
    print(f"Valid files       : {len(valid_records)}")
    print(f"Invalid files     : {len(invalid_files)}")
    print(f"Unique SHA-256    : {len(hash_to_files)}")
    print(f"Duplicate clusters: {len(duplicate_clusters)}")

    print()
    print("Class counts:")
    for genre, count in sorted(class_counts.items()):
        print(f"  {genre:20s} {count:4d}")

    print()
    print("Sample rates:")
    for sr, count in sorted(sample_rate_counts.items()):
        print(f"  {sr:6d} Hz : {count}")

    print()
    print("Channels:")
    for channels, count in sorted(channel_counts.items()):
        print(f"  {channels} channel(s) : {count}")

    if duration_stats:
        print()
        print("Duration statistics:")
        for key, value in duration_stats.items():
            print(f"  {key:20s}: {value:.2f}")

    print()
    print(f"Files < 2 sec       : {len(short_files)}")
    print(f"Files > 10 min      : {len(long_files)}")
    print(f"Silent files        : {len(silent_files)}")
    print(f"Low-energy files    : {len(low_energy_files)}")

    print()
    print("Outputs:")
    print(f"  {CSV_PATH}")
    print(f"  {JSON_PATH}")


if __name__ == "__main__":
    main()