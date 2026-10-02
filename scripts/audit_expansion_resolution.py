"""
GENGROGRAM - Expanded Dataset Resolution Audit

Purpose:
    Resolve the issues found during the expansion audio audit.

Checks:
    1. Invalid/unreadable expansion files
    2. Very short files
    3. Silent / low-energy files
    4. Exact SHA-256 duplicate clusters
    5. Cross-genre duplicate clusters
    6. Cross-dataset duplicates against GTZAN

IMPORTANT:
    READ-ONLY.
    This script does not modify or delete any audio files.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPANSION_ROOT = PROJECT_ROOT / "data" / "expanded_raw"
GTZAN_ROOT = PROJECT_ROOT / "data" / "gtzan" / "genres"

AUDIT_ROOT = PROJECT_ROOT / "audit_outputs" / "expansion"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}

AUDIO_INVENTORY = AUDIT_ROOT / "audio_inventory.csv"
OUTPUT_JSON = AUDIT_ROOT / "resolution_audit.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def class_from_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)

    if len(relative.parts) < 2:
        return "UNKNOWN"

    return relative.parts[0]


def safe_load_audio(path: Path):
    """
    Load audio only for inspection.
    No modification is performed.
    """
    try:
        y, sr = librosa.load(
            str(path),
            sr=None,
            mono=True,
        )

        return y, sr, None

    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def rms_db(y: np.ndarray) -> float:
    if y is None or y.size == 0:
        return -120.0

    rms = float(
        np.sqrt(
            np.mean(
                np.square(y),
                dtype=np.float64,
            )
        )
    )

    if rms <= 0:
        return -120.0

    return float(20.0 * np.log10(rms))


def collect_expansion_files():
    return sorted(
        p
        for p in EXPANSION_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def collect_gtzan_files():
    if not GTZAN_ROOT.exists():
        return []

    return sorted(
        p
        for p in GTZAN_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def main():

    print("=" * 72)
    print("GENGROGRAM - EXPANDED DATASET RESOLUTION AUDIT")
    print("=" * 72)

    print(f"Expansion root : {EXPANSION_ROOT}")
    print(f"GTZAN root     : {GTZAN_ROOT}")
    print()

    if not EXPANSION_ROOT.exists():
        raise FileNotFoundError(
            f"Expansion dataset not found: {EXPANSION_ROOT}"
        )

    expansion_files = collect_expansion_files()

    print(
        f"Expansion audio files: {len(expansion_files)}"
    )
    print()

    # ------------------------------------------------------------
    # 1. Inspect problematic files
    # ------------------------------------------------------------

    invalid_files = []
    short_files = []
    silent_files = []
    low_energy_files = []

    expansion_hashes = defaultdict(list)

    file_details = {}

    print("Inspecting expansion files...")

    for index, path in enumerate(expansion_files, start=1):

        relative = path.relative_to(EXPANSION_ROOT)
        genre = class_from_path(EXPANSION_ROOT, path)

        y, sr, error = safe_load_audio(path)

        detail = {
            "relative_path": str(relative),
            "genre": genre,
            "sample_rate": sr,
            "duration_seconds": None,
            "rms_db": None,
            "sha256": None,
            "error": error,
        }

        if error is not None:
            invalid_files.append(detail)
            file_details[str(relative)] = detail
            continue

        duration = (
            float(len(y)) / float(sr)
            if sr and sr > 0
            else 0.0
        )

        level = rms_db(y)

        detail["duration_seconds"] = duration
        detail["rms_db"] = level

        if duration < 2.0:
            short_files.append(detail)

        if level <= -80.0:
            silent_files.append(detail)

        if level <= -45.0:
            low_energy_files.append(detail)

        file_hash = sha256_file(path)

        detail["sha256"] = file_hash

        expansion_hashes[file_hash].append(
            str(relative)
        )

        file_details[str(relative)] = detail

        if index % 100 == 0 or index == len(expansion_files):
            print(
                f"  [{index:4d}/{len(expansion_files)}]"
            )

    # ------------------------------------------------------------
    # 2. Exact duplicate clusters
    # ------------------------------------------------------------

    duplicate_clusters = []

    for file_hash, paths in sorted(
        expansion_hashes.items()
    ):
        if len(paths) > 1:

            genres = sorted(
                {
                    class_from_path(
                        EXPANSION_ROOT,
                        EXPANSION_ROOT / p
                    )
                    for p in paths
                }
            )

            duplicate_clusters.append(
                {
                    "sha256": file_hash,
                    "count": len(paths),
                    "genres": genres,
                    "cross_genre": len(genres) > 1,
                    "files": paths,
                }
            )

    cross_genre_duplicates = [
        cluster
        for cluster in duplicate_clusters
        if cluster["cross_genre"]
    ]

    # ------------------------------------------------------------
    # 3. GTZAN cross-dataset hash audit
    # ------------------------------------------------------------

    print()
    print("Checking GTZAN cross-dataset duplicates...")

    gtzan_files = collect_gtzan_files()

    print(
        f"GTZAN audio files discovered: {len(gtzan_files)}"
    )

    gtzan_hashes = defaultdict(list)

    for index, path in enumerate(gtzan_files, start=1):

        try:
            file_hash = sha256_file(path)

            gtzan_hashes[file_hash].append(
                str(path.relative_to(GTZAN_ROOT))
            )

        except Exception:
            pass

        if index % 100 == 0 or index == len(gtzan_files):
            print(
                f"  [{index:4d}/{len(gtzan_files)}]"
            )

    cross_dataset_duplicates = []

    for file_hash, expansion_paths in expansion_hashes.items():

        if file_hash in gtzan_hashes:

            cross_dataset_duplicates.append(
                {
                    "sha256": file_hash,
                    "expansion_files": expansion_paths,
                    "gtzan_files": gtzan_hashes[file_hash],
                }
            )

    # ------------------------------------------------------------
    # 4. Produce summary
    # ------------------------------------------------------------

    result = {
        "expansion_total_files": len(expansion_files),

        "invalid_files": {
            "count": len(invalid_files),
            "files": invalid_files,
        },

        "short_files_under_2_seconds": {
            "count": len(short_files),
            "files": short_files,
        },

        "silent_files_rms_le_minus_80_db": {
            "count": len(silent_files),
            "files": silent_files,
        },

        "low_energy_files_rms_le_minus_45_db": {
            "count": len(low_energy_files),
            "files": low_energy_files,
        },

        "exact_duplicate_clusters": {
            "count": len(duplicate_clusters),
            "clusters": duplicate_clusters,
        },

        "cross_genre_exact_duplicates": {
            "count": len(cross_genre_duplicates),
            "clusters": cross_genre_duplicates,
        },

        "gtzan_total_files": len(gtzan_files),

        "cross_dataset_exact_duplicates": {
            "count": len(cross_dataset_duplicates),
            "matches": cross_dataset_duplicates,
        },
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # 5. Console report
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("RESOLUTION AUDIT COMPLETE")
    print("=" * 72)

    print()
    print("Problem files")
    print("-" * 72)

    print(
        f"Invalid files       : {len(invalid_files)}"
    )
    print(
        f"Files < 2 sec       : {len(short_files)}"
    )
    print(
        f"Silent files        : {len(silent_files)}"
    )
    print(
        f"Low-energy files    : {len(low_energy_files)}"
    )

    if invalid_files:
        print()
        print("INVALID FILES:")
        for item in invalid_files:
            print(
                f"  {item['relative_path']}"
            )
            print(
                f"    ERROR: {item['error']}"
            )

    if short_files:
        print()
        print("SHORT FILES (<2 sec):")

        for item in short_files:
            print(
                f"  {item['duration_seconds']:.3f}s "
                f"| {item['genre']} "
                f"| {item['relative_path']}"
            )

    if silent_files:
        print()
        print("SILENT FILES:")

        for item in silent_files:
            print(
                f"  {item['rms_db']:.2f} dB "
                f"| {item['genre']} "
                f"| {item['relative_path']}"
            )

    if low_energy_files:
        print()
        print("LOW-ENERGY FILES:")

        for item in low_energy_files:
            print(
                f"  {item['rms_db']:.2f} dB "
                f"| {item['genre']} "
                f"| {item['relative_path']}"
            )

    print()
    print("Exact duplicate clusters")
    print("-" * 72)

    print(
        f"Total duplicate clusters : "
        f"{len(duplicate_clusters)}"
    )

    print(
        f"Cross-genre clusters     : "
        f"{len(cross_genre_duplicates)}"
    )

    for index, cluster in enumerate(
        duplicate_clusters,
        start=1,
    ):

        print()
        print(
            f"Cluster {index} "
            f"({cluster['count']} files)"
        )

        print(
            f"Genres: {', '.join(cluster['genres'])}"
        )

        for file_path in cluster["files"]:
            print(
                f"  {file_path}"
            )

    print()
    print("GTZAN cross-dataset duplicate check")
    print("-" * 72)

    print(
        f"GTZAN files checked       : "
        f"{len(gtzan_files)}"
    )

    print(
        f"Exact cross-dataset matches: "
        f"{len(cross_dataset_duplicates)}"
    )

    if cross_dataset_duplicates:

        for match in cross_dataset_duplicates:

            print()
            print(
                f"SHA256: {match['sha256']}"
            )

            print("Expansion:")
            for p in match["expansion_files"]:
                print(f"  {p}")

            print("GTZAN:")
            for p in match["gtzan_files"]:
                print(f"  {p}")

    print()
    print("Output:")
    print(OUTPUT_JSON)

    print()
    print(
        "No raw audio files were modified."
    )


if __name__ == "__main__":
    main()