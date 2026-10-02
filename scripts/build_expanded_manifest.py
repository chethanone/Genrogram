"""
GENGROGRAM - Expanded 20-Class Dataset Manifest Builder

Purpose:
    Build a clean, auditable manifest for the expanded 20-class
    genre-classification dataset.

IMPORTANT:
    - Raw datasets are READ-ONLY.
    - No audio files are moved, renamed, copied, converted, or deleted.
    - The manifest records which files are eligible for training.
    - Exact duplicate recordings are represented only once.
    - Cross-label duplicate recordings are assigned one canonical label.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GTZAN_ROOT = PROJECT_ROOT / "data" / "gtzan" / "genres"
EXPANSION_ROOT = PROJECT_ROOT / "data" / "expanded_raw"

AUDIT_ROOT = PROJECT_ROOT / "audit_outputs" / "expansion"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "manifests"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUTPUT_ROOT / "expanded_20class_manifest.csv"
SUMMARY_JSON = OUTPUT_ROOT / "expanded_20class_manifest_summary.json"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


# ------------------------------------------------------------
# Canonical 20-class taxonomy
# ------------------------------------------------------------

GTZAN_LABELS = {
    "blues": "Blues",
    "classical": "Classical",
    "country": "Country",
    "disco": "Disco",
    "hiphop": "Hip-Hop",
    "jazz": "Jazz",
    "metal": "Metal",
    "pop": "Pop",
    "reggae": "Reggae",
    "rock": "Rock",
}

EXPANSION_LABELS = {
    "Amapiano": "Amapiano",
    "Hyperpop": "Hyperpop",
    "K-Pop": "K-Pop",
    "Phonk": "Phonk",
    "Techno": "Techno",
    "Bollywood": "Bollywood",
    "Desi_Hip-Hop": "Desi Hip-Hop",
    "Haryanvi": "Haryanvi",
    "I-Pop": "I-Pop",
    "Punjabi_Pop": "Punjabi Pop",
}


# ------------------------------------------------------------
# Canonical decisions for cross-label exact duplicates
#
# Rule:
#     Keep ONE canonical label and exclude the duplicate copy
#     under the competing label.
#
# This does NOT delete the raw files.
# ------------------------------------------------------------

CROSS_LABEL_CANONICAL = {
    "Brown_Munde": "Desi Hip-Hop",
    "Cheques_Music_Video": "Desi Hip-Hop",
    "On_Top": "Desi Hip-Hop",
    "indi_aura": "Desi Hip-Hop",
    "Karan_Aujla": "Desi Hip-Hop",

    "HUNTRX_-_Golden": "K-Pop",
    "Takedown_Song": "K-Pop",
    "Soda_Pop_Saja": "K-Pop",
}


# ------------------------------------------------------------
# Files known to be silent/unusable from resolution audit
# ------------------------------------------------------------

EXCLUDE_SILENT_SUBSTRINGS = [
    "Freakin_Music_Entertainment_-_The_Bads_of_Bollywood",
    "DesiHipHop_-_Trapt_Saheer",
    "Irshad_Khan_-_Haryana_Hood",
    "J_Singh_-_J_Singh_Dard_Rap",
    "Kluz_Cartoon_-_Your_Idol_Not_Official",
    "SilentHum_-_Saja_Boys_Golden",
    "VIBMUSIC_-_TOP_10_MOST_VIRAL_PHONKFUNK_2025",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def is_silent_excluded(path: Path) -> bool:
    name = path.name

    return any(
        token.lower() in name.lower()
        for token in EXCLUDE_SILENT_SUBSTRINGS
    )


def get_audio_duration(path: Path) -> float:
    """
    Read duration without converting or modifying the source.
    """
    try:
        y, sr = librosa.load(
            str(path),
            sr=None,
            mono=True,
        )

        if sr <= 0:
            return 0.0

        return float(len(y) / sr)

    except Exception:
        return 0.0


def is_audio_readable(path: Path) -> bool:
    """
    Verify that the audio file can actually be decoded.

    This is a read-only validation check.
    It does not modify the source file.
    """
    try:
        y, sr = librosa.load(
            str(path),
            sr=None,
            mono=True,
        )

        if sr <= 0:
            return False

        if y is None or len(y) == 0:
            return False

        return bool(
            np.isfinite(y).all()
        )

    except Exception:
        return False


def add_reason(
    record: dict,
    reason: str,
):
    record["status"] = "excluded"
    record["exclusion_reason"] = reason


def collect_files(root: Path):
    """
    Collect genuine audio candidates only.

    AppleDouble files such as ._filename.wav are macOS
    filesystem metadata, not audio recordings.

    Raw files are not modified or deleted.
    """
    if not root.exists():
        return []

    return sorted(
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and not p.name.startswith("._")
            and not p.name.startswith(".")
            and p.suffix.lower() in AUDIO_EXTENSIONS
        )
    )


def build_gtzan_records():
    records = []

    files = collect_files(GTZAN_ROOT)

    print(
        f"GTZAN files discovered: {len(files)}"
    )

    for path in files:

        relative = path.relative_to(GTZAN_ROOT)

        if not relative.parts:
            continue

        source_label = relative.parts[0].lower()

        if source_label not in GTZAN_LABELS:
            continue

        genre = GTZAN_LABELS[source_label]

        record = {
            "source": "GTZAN",
            "source_class": source_label,
            "genre": genre,
            "relative_path": str(relative),
            "absolute_path": str(path),
            "sha256": "",
            "duration_seconds": 0.0,
            "status": "included",
            "exclusion_reason": "",
        }

        # Reject files that cannot actually be decoded.
        if not is_audio_readable(path):
            add_reason(
                record,
                "unreadable_or_corrupt_audio",
            )

            records.append(record)
            continue

        file_hash = sha256_file(path)
        duration = get_audio_duration(path)

        record["sha256"] = file_hash
        record["duration_seconds"] = duration

        if duration <= 0.0:
            add_reason(
                record,
                "zero_duration",
            )

        records.append(record)

        records.append(
            {
                "source": "GTZAN",
                "source_class": source_label,
                "genre": genre,
                "relative_path": str(relative),
                "absolute_path": str(path),
                "sha256": file_hash,
                "duration_seconds": duration,
                "status": "included",
                "exclusion_reason": "",
            }
        )

    return records


def build_expansion_records():
    records = []

    files = collect_files(EXPANSION_ROOT)

    print(
        f"Expansion files discovered: {len(files)}"
    )

    for path in files:

        relative = path.relative_to(EXPANSION_ROOT)

        if not relative.parts:
            continue

        source_class = relative.parts[0]

        if source_class not in EXPANSION_LABELS:
            continue

        genre = EXPANSION_LABELS[source_class]

        file_hash = sha256_file(path)

        duration = get_audio_duration(path)

        record = {
            "source": "Navrasa",
            "source_class": source_class,
            "genre": genre,
            "relative_path": str(relative),
            "absolute_path": str(path),
            "sha256": file_hash,
            "duration_seconds": duration,
            "status": "included",
            "exclusion_reason": "",
        }

        if not is_audio_readable(path):
            add_reason(
                record,
                "unreadable_or_corrupt_audio",
            )

            records.append(record)
            continue        

        # Explicitly exclude the objectively silent files.
        elif is_silent_excluded(path):
            add_reason(
                record,
                "silent_or_zero_audio",
            )

        # Zero-duration recordings are unusable.
        elif duration <= 0.0:
            add_reason(
                record,
                "zero_duration",
            )

        records.append(record)

    return records


def canonical_duplicate_key(record: dict) -> str:
    """
    Produce a normalized key from filename for the small number
    of known cross-label duplicate recordings.

    Exact SHA-256 is always checked first. This key is only used
    to make the canonical-label decision explicit.
    """

    name = Path(
        record["relative_path"]
    ).stem

    return name


def resolve_duplicate_groups(records):
    """
    Exact-hash duplicate resolution.

    GTZAN is retained as-is.

    Within the expansion dataset:
        - same-label duplicate -> retain first canonical copy
        - cross-label duplicate -> use canonical label policy
    """

    hash_groups = defaultdict(list)

    for record in records:
        hash_groups[record["sha256"]].append(record)

    duplicate_groups = []

    for file_hash, group in hash_groups.items():

        if len(group) <= 1:
            continue

        duplicate_groups.append(
            {
                "sha256": file_hash,
                "records": group,
            }
        )

    # Process only expansion duplicate groups.
    for group in duplicate_groups:

        group_records = group["records"]

        expansion_records = [
            r
            for r in group_records
            if r["source"] == "Navrasa"
            and r["status"] == "included"
        ]

        if len(expansion_records) <= 1:
            continue

        genres = {
            r["genre"]
            for r in expansion_records
        }

        if len(genres) == 1:

            # Same-label duplicate:
            # keep deterministic first path.
            sorted_records = sorted(
                expansion_records,
                key=lambda r: r["relative_path"],
            )

            keeper = sorted_records[0]

            for duplicate in sorted_records[1:]:
                add_reason(
                    duplicate,
                    "exact_duplicate_same_class",
                )

        else:

            # Cross-label duplicate.
            #
            # Determine canonical class from filename/path.
            canonical_record = None

            for candidate in expansion_records:

                key = canonical_duplicate_key(
                    candidate
                )

                for token, canonical_genre in (
                    CROSS_LABEL_CANONICAL.items()
                ):

                    if token.lower() in key.lower():
                        if (
                            candidate["genre"]
                            == canonical_genre
                        ):
                            canonical_record = candidate
                            break

                if canonical_record is not None:
                    break

            # If a known canonical label was not found,
            # fall back to deterministic lexical order.
            if canonical_record is None:
                canonical_record = sorted(
                    expansion_records,
                    key=lambda r: (
                        r["genre"],
                        r["relative_path"],
                    ),
                )[0]

            for candidate in expansion_records:

                if candidate is canonical_record:
                    continue

                add_reason(
                    candidate,
                    (
                        "exact_duplicate_cross_class;"
                        f"canonical_class="
                        f"{canonical_record['genre']}"
                    ),
                )

    return duplicate_groups


def resolve_cross_dataset_hashes(
    records,
):
    """
    Defensive cross-source duplicate check.

    GTZAN and expansion were already checked separately.
    This performs the same check while constructing the
    final manifest.
    """

    gtzan_hashes = {
        r["sha256"]
        for r in records
        if r["source"] == "GTZAN"
    }

    cross_matches = []

    for record in records:

        if record["source"] != "Navrasa":
            continue

        if record["sha256"] in gtzan_hashes:

            add_reason(
                record,
                "exact_duplicate_with_gtzan",
            )

            cross_matches.append(
                record["relative_path"]
            )

    return cross_matches


def main():

    print("=" * 72)
    print("GENGROGRAM - BUILD EXPANDED 20-CLASS MANIFEST")
    print("=" * 72)
    print()

    # ------------------------------------------------------------
    # Build records
    # ------------------------------------------------------------

    print("Building GTZAN records...")
    gtzan_records = build_gtzan_records()

    print()
    print("Building expansion records...")
    expansion_records = build_expansion_records()

    records = (
        gtzan_records
        + expansion_records
    )

    print()
    print(
        f"Total candidate records: {len(records)}"
    )

    # ------------------------------------------------------------
    # Duplicate resolution
    # ------------------------------------------------------------

    print()
    print("Resolving exact duplicate groups...")

    duplicate_groups = resolve_duplicate_groups(
        records
    )

    print(
        f"Duplicate groups found: "
        f"{len(duplicate_groups)}"
    )

    # ------------------------------------------------------------
    # Cross-source protection
    # ------------------------------------------------------------

    print()
    print("Checking GTZAN/expansion cross-source hashes...")

    cross_dataset_matches = (
        resolve_cross_dataset_hashes(records)
    )

    print(
        f"Cross-source exact matches: "
        f"{len(cross_dataset_matches)}"
    )

    # ------------------------------------------------------------
    # Final included records
    # ------------------------------------------------------------

    included = [
        r
        for r in records
        if r["status"] == "included"
    ]

    excluded = [
        r
        for r in records
        if r["status"] != "included"
    ]

    # Deterministic ordering.
    included.sort(
        key=lambda r: (
            r["genre"],
            r["source"],
            r["relative_path"],
        )
    )

    excluded.sort(
        key=lambda r: (
            r["genre"],
            r["source"],
            r["relative_path"],
        )
    )

    # ------------------------------------------------------------
    # Class statistics
    # ------------------------------------------------------------

    included_counts = Counter(
        r["genre"]
        for r in included
    )

    excluded_counts = Counter(
        r["genre"]
        for r in excluded
    )

    source_counts = Counter(
        r["source"]
        for r in included
    )

    exclusion_reasons = Counter(
        r["exclusion_reason"]
        for r in excluded
    )

    # ------------------------------------------------------------
    # Class IDs
    # ------------------------------------------------------------

    ordered_genres = [
        "Blues",
        "Classical",
        "Country",
        "Disco",
        "Hip-Hop",
        "Jazz",
        "Metal",
        "Pop",
        "Reggae",
        "Rock",
        "Amapiano",
        "Hyperpop",
        "K-Pop",
        "Phonk",
        "Techno",
        "Bollywood",
        "Desi Hip-Hop",
        "Haryanvi",
        "I-Pop",
        "Punjabi Pop",
    ]

    class_mapping = {
        genre: index
        for index, genre in enumerate(
            ordered_genres
        )
    }

    # ------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------

    fieldnames = [
        "sample_id",
        "source",
        "source_class",
        "genre",
        "class_id",
        "relative_path",
        "absolute_path",
        "sha256",
        "duration_seconds",
        "status",
        "exclusion_reason",
    ]

    with MANIFEST_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for sample_id, record in enumerate(
            included,
            start=1,
        ):

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source": record["source"],
                    "source_class": record["source_class"],
                    "genre": record["genre"],
                    "class_id": class_mapping[
                        record["genre"]
                    ],
                    "relative_path": record[
                        "relative_path"
                    ],
                    "absolute_path": record[
                        "absolute_path"
                    ],
                    "sha256": record["sha256"],
                    "duration_seconds": (
                        f"{record['duration_seconds']:.6f}"
                    ),
                    "status": record["status"],
                    "exclusion_reason": record[
                        "exclusion_reason"
                    ],
                }
            )

    # ------------------------------------------------------------
    # Write summary JSON
    # ------------------------------------------------------------

    summary = {
        "dataset_name": (
            "GENGROGRAM Expanded 20-Class Dataset"
        ),

        "total_candidate_records": len(records),

        "included_records": len(included),

        "excluded_records": len(excluded),

        "source_counts": dict(
            sorted(source_counts.items())
        ),

        "class_counts": {
            genre: included_counts.get(
                genre,
                0,
            )
            for genre in ordered_genres
        },

        "excluded_class_counts": {
            genre: excluded_counts.get(
                genre,
                0,
            )
            for genre in ordered_genres
        },

        "exclusion_reasons": dict(
            sorted(exclusion_reasons.items())
        ),

        "class_mapping": class_mapping,

        "duplicate_group_count": len(
            duplicate_groups
        ),

        "cross_dataset_exact_matches": len(
            cross_dataset_matches
        ),

        "manifest_csv": str(MANIFEST_CSV),

        "raw_data_unchanged": True,

        "policy": {
            "raw_audio_modified": False,
            "silent_files_excluded": True,
            "exact_duplicates_collapsed": True,
            "cross_class_duplicates_resolved": True,
            "gtzan_cross_source_duplicates_checked": True,
        },
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("MANIFEST BUILD COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Candidate records : {len(records)}"
    )
    print(
        f"Included records  : {len(included)}"
    )
    print(
        f"Excluded records  : {len(excluded)}"
    )

    print()
    print("FINAL CLASS COUNTS")
    print("-" * 72)

    for genre in ordered_genres:

        count = included_counts.get(
            genre,
            0,
        )

        print(
            f"{genre:20s} : {count:4d}"
        )

    print()
    print("EXCLUSION REASONS")
    print("-" * 72)

    for reason, count in sorted(
        exclusion_reasons.items()
    ):

        print(
            f"{reason:45s} : {count:4d}"
        )

    print()
    print("SOURCE COUNTS")
    print("-" * 72)

    for source, count in sorted(
        source_counts.items()
    ):

        print(
            f"{source:20s} : {count:4d}"
        )

    print()
    print("OUTPUTS")
    print("-" * 72)

    print(MANIFEST_CSV)
    print(SUMMARY_JSON)

    print()
    print(
        "Raw datasets were not modified."
    )

    print()
    print(
        "DO NOT TRAIN YET."
    )
    print(
        "Review the final class counts before "
        "creating the leakage-safe split."
    )


if __name__ == "__main__":
    main()