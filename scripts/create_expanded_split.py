"""
GENGROGRAM - Leakage-Safe 20-Class Dataset Split

Creates:
    70% train
    15% validation
    15% test

Properties:
    - Track-level splitting
    - SHA-256 duplicate groups kept together
    - Stratified by genre
    - Deterministic
    - Raw audio untouched
    - Existing frozen manifest is not modified
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "expanded_20class_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "manifests"
)

TRAIN_PATH = OUTPUT_ROOT / "train_manifest.csv"
VAL_PATH = OUTPUT_ROOT / "validation_manifest.csv"
TEST_PATH = OUTPUT_ROOT / "test_manifest.csv"
SUMMARY_PATH = OUTPUT_ROOT / "split_summary.json"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def read_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
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

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def make_groups(rows):
    """
    Every exact SHA-256 hash is an indivisible group.

    In normal circumstances most groups contain one track.
    Existing exact duplicates remain together.
    """

    groups = defaultdict(list)

    for row in rows:
        groups[row["sha256"]].append(row)

    return list(groups.values())


def allocate_groups(groups, seed):
    """
    Allocate SHA-256 duplicate groups to train/validation/test.

    Important:
        A duplicate group may contain multiple genre labels,
        particularly in the original GTZAN dataset.

        Such a group is treated as ONE indivisible unit so that
        identical audio cannot cross train/validation/test.

    The algorithm greedily balances genre counts while keeping
    every SHA-256 group intact.
    """

    rng = random.Random(seed)

    # ------------------------------------------------------------
    # Calculate total class counts.
    # ------------------------------------------------------------

    total_class_counts = Counter()

    for group in groups:
        for row in group:
            total_class_counts[row["genre"]] += 1

    # Desired approximate counts.
    desired = {
        "train": {},
        "validation": {},
        "test": {},
    }

    for genre, total in total_class_counts.items():

        desired["train"][genre] = total * TRAIN_RATIO
        desired["validation"][genre] = total * VAL_RATIO
        desired["test"][genre] = total * TEST_RATIO

    # ------------------------------------------------------------
    # Shuffle groups first so equal-sized candidates don't always
    # receive the same assignment.
    # ------------------------------------------------------------

    shuffled_groups = list(groups)

    rng.shuffle(shuffled_groups)

    # Process larger / more complex groups first.
    #
    # A group containing many records and/or many genres has a
    # larger effect on the class distribution and should therefore
    # be allocated before ordinary single-record groups.
    shuffled_groups.sort(
        key=lambda group: (
            len(group),
            len({
                row["genre"]
                for row in group
            }),
        ),
        reverse=True,
    )

    split_groups = {
        "train": [],
        "validation": [],
        "test": [],
    }

    current_counts = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    # ------------------------------------------------------------
    # Score a possible assignment.
    # ------------------------------------------------------------

    def assignment_score(
        group,
        split_name,
    ):
        score = 0.0

        for row in group:

            genre = row["genre"]

            before = current_counts[
                split_name
            ][genre]

            after = before + 1

            target = desired[
                split_name
            ][genre]

            # Penalize going beyond the target.
            if after > target:
                score += (
                    after - target
                ) * 10.0

            # Reward filling an underrepresented class.
            remaining = max(
                target - before,
                0.0,
            )

            score -= remaining * 0.001

        # Small global size penalty.
        current_total = sum(
            len(g)
            for g in split_groups[
                split_name
            ]
        )

        desired_total = (
            len(groups)
            * {
                "train": TRAIN_RATIO,
                "validation": VAL_RATIO,
                "test": TEST_RATIO,
            }[split_name]
        )

        projected_total = (
            current_total + len(group)
        )

        score += abs(
            projected_total
            - desired_total
        ) * 0.001

        return score

    # ------------------------------------------------------------
    # Assign each group.
    # ------------------------------------------------------------

    for group in shuffled_groups:

        scores = {
            split_name: assignment_score(
                group,
                split_name,
            )
            for split_name in (
                "train",
                "validation",
                "test",
            )
        }

        best_score = min(
            scores.values()
        )

        best_splits = [
            split_name
            for split_name, score
            in scores.items()
            if abs(score - best_score)
            < 1e-9
        ]

        selected = rng.choice(
            best_splits
        )

        split_groups[
            selected
        ].append(group)

        for row in group:

            current_counts[
                selected
            ][row["genre"]] += 1

    # ------------------------------------------------------------
    # Flatten groups.
    # ------------------------------------------------------------

    train = [
        row
        for group in split_groups["train"]
        for row in group
    ]

    validation = [
        row
        for group in split_groups["validation"]
        for row in group
    ]

    test = [
        row
        for group in split_groups["test"]
        for row in group
    ]

    # ------------------------------------------------------------
    # Statistics.
    # ------------------------------------------------------------

    split_stats = {}

    for genre in sorted(
        total_class_counts
    ):

        split_stats[genre] = {
            "total_records": total_class_counts[
                genre
            ],
            "train_records": current_counts[
                "train"
            ][genre],
            "validation_records": current_counts[
                "validation"
            ][genre],
            "test_records": current_counts[
                "test"
            ][genre],
        }

    return (
        split_groups["train"],
        split_groups["validation"],
        split_groups["test"],
        split_stats,
    )


def flatten(groups):
    rows = []

    for group in groups:
        rows.extend(group)

    return rows


def hash_set(rows):
    return {
        row["sha256"]
        for row in rows
    }


def path_set(rows):
    return {
        row["relative_path"]
        for row in rows
    }


def class_counts(rows):
    return dict(
        sorted(
            Counter(
                row["genre"]
                for row in rows
            ).items()
        )
    )


def source_counts(rows):
    return dict(
        sorted(
            Counter(
                row["source"]
                for row in rows
            ).items()
        )
    )


def main():

    print("=" * 72)
    print("GENGROGRAM - LEAKAGE-SAFE 20-CLASS SPLIT")
    print("=" * 72)
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    rows = read_manifest()

    print(
        f"Manifest records: {len(rows)}"
    )

    if not rows:
        raise ValueError(
            "Manifest is empty."
        )

    # ------------------------------------------------------------
    # Duplicate groups
    # ------------------------------------------------------------

    groups = make_groups(rows)

    print(
        f"SHA-256 groups: {len(groups)}"
    )

    duplicate_group_count = sum(
        1
        for group in groups
        if len(group) > 1
    )

    duplicate_record_count = sum(
        len(group)
        for group in groups
        if len(group) > 1
    )

    print(
        f"Duplicate groups: "
        f"{duplicate_group_count}"
    )

    print(
        f"Records inside duplicate groups: "
        f"{duplicate_record_count}"
    )

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    (
        train_groups,
        val_groups,
        test_groups,
        split_stats,
    ) = allocate_groups(
        groups,
        SEED,
    )

    train = flatten(train_groups)
    validation = flatten(val_groups)
    test = flatten(test_groups)

    # ------------------------------------------------------------
    # Deterministic row ordering
    # ------------------------------------------------------------

    sort_key = lambda row: (
        int(row["class_id"]),
        row["genre"],
        row["relative_path"],
    )

    train.sort(key=sort_key)
    validation.sort(key=sort_key)
    test.sort(key=sort_key)

    # ------------------------------------------------------------
    # Verify every record appears exactly once
    # ------------------------------------------------------------

    all_paths = (
        path_set(train)
        | path_set(validation)
        | path_set(test)
    )

    original_paths = path_set(rows)

    if all_paths != original_paths:
        missing = original_paths - all_paths
        extra = all_paths - original_paths

        raise RuntimeError(
            "Split does not contain exactly the original "
            f"records.\nMissing={missing}\nExtra={extra}"
        )

    if (
        len(train)
        + len(validation)
        + len(test)
        != len(rows)
    ):
        raise RuntimeError(
            "Split record counts do not add up."
        )

    # ------------------------------------------------------------
    # Leakage verification
    # ------------------------------------------------------------

    train_hashes = hash_set(train)
    val_hashes = hash_set(validation)
    test_hashes = hash_set(test)

    train_val_overlap = (
        train_hashes & val_hashes
    )

    train_test_overlap = (
        train_hashes & test_hashes
    )

    val_test_overlap = (
        val_hashes & test_hashes
    )

    if train_val_overlap:
        raise RuntimeError(
            "TRAIN/VALIDATION SHA-256 LEAKAGE DETECTED"
        )

    if train_test_overlap:
        raise RuntimeError(
            "TRAIN/TEST SHA-256 LEAKAGE DETECTED"
        )

    if val_test_overlap:
        raise RuntimeError(
            "VALIDATION/TEST SHA-256 LEAKAGE DETECTED"
        )

    # ------------------------------------------------------------
    # Write split manifests
    # ------------------------------------------------------------

    write_manifest(
        TRAIN_PATH,
        train,
    )

    write_manifest(
        VAL_PATH,
        validation,
    )

    write_manifest(
        TEST_PATH,
        test,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {
        "dataset": "GENGROGRAM Expanded 20-Class Dataset",

        "seed": SEED,

        "ratios_requested": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },

        "total_records": len(rows),

        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },

        "split_percentages": {
            "train": len(train) / len(rows),
            "validation": len(validation) / len(rows),
            "test": len(test) / len(rows),
        },

        "class_counts": {
            "train": class_counts(train),
            "validation": class_counts(
                validation
            ),
            "test": class_counts(test),
        },

        "source_counts": {
            "train": source_counts(train),
            "validation": source_counts(
                validation
            ),
            "test": source_counts(test),
        },

        "duplicate_group_statistics": {
            "total_sha256_groups": len(groups),
            "duplicate_group_count": (
                duplicate_group_count
            ),
            "records_inside_duplicate_groups": (
                duplicate_record_count
            ),
        },

        "leakage_check": {
            "train_validation_sha256_overlap": (
                len(train_val_overlap)
            ),
            "train_test_sha256_overlap": (
                len(train_test_overlap)
            ),
            "validation_test_sha256_overlap": (
                len(val_test_overlap)
            ),
            "passed": (
                not train_val_overlap
                and not train_test_overlap
                and not val_test_overlap
            ),
        },

        "per_class_group_split": split_stats,

        "outputs": {
            "train": str(TRAIN_PATH),
            "validation": str(VAL_PATH),
            "test": str(TEST_PATH),
        },
    }

    with SUMMARY_PATH.open(
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
    print("SPLIT COMPLETE")
    print("=" * 72)

    print()
    print("OVERALL")
    print("-" * 72)

    print(
        f"Train      : {len(train):4d} "
        f"({len(train) / len(rows) * 100:.2f}%)"
    )

    print(
        f"Validation : {len(validation):4d} "
        f"({len(validation) / len(rows) * 100:.2f}%)"
    )

    print(
        f"Test       : {len(test):4d} "
        f"({len(test) / len(rows) * 100:.2f}%)"
    )

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 72)

    genres = sorted(
        set(row["genre"] for row in rows),
        key=lambda g: int(
            next(
                row["class_id"]
                for row in rows
                if row["genre"] == g
            )
        ),
    )

    for genre in genres:

        tr = sum(
            1
            for row in train
            if row["genre"] == genre
        )

        va = sum(
            1
            for row in validation
            if row["genre"] == genre
        )

        te = sum(
            1
            for row in test
            if row["genre"] == genre
        )

        print(
            f"{genre:20s} "
            f"Train={tr:3d}  "
            f"Val={va:3d}  "
            f"Test={te:3d}"
        )

    print()
    print("SOURCE DISTRIBUTION")
    print("-" * 72)

    for split_name, split_rows in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        counts = source_counts(
            split_rows
        )

        print(
            f"{split_name:10s} : "
            + " | ".join(
                f"{k}={v}"
                for k, v in counts.items()
            )
        )

    print()
    print("LEAKAGE CHECK")
    print("-" * 72)

    print(
        f"Train ↔ Validation : "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ↔ Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ↔ Test  : "
        f"{len(val_test_overlap)}"
    )

    print()
    print(
        "RESULT: "
        + (
            "PASS - no SHA-256 leakage detected."
            if (
                not train_val_overlap
                and not train_test_overlap
                and not val_test_overlap
            )
            else
            "FAIL - leakage detected."
        )
    )

    print()
    print("OUTPUTS")
    print("-" * 72)

    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)
    print(SUMMARY_PATH)

    print()
    print(
        "Raw audio and the frozen master manifest "
        "were not modified."
    )


if __name__ == "__main__":
    main()