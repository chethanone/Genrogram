"""Audit the local expanded-genre raw folders without modifying them.

Expected layout:
  data/expanded_raw/gtzan/genres/<genre>/*
  data/expanded_raw/navrasa/<genre>/*

The script records audio metadata and SHA-256 hashes and reports class counts,
invalid files, duplicates, and duration statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def audit(root: Path, output_csv: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("._") or path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        source = parts[0] if len(parts) >= 2 else "unknown"
        genre = parts[1] if len(parts) >= 3 else (path.parent.name if path.parent != root else "unknown")
        row = {
            "source": source,
            "genre": genre,
            "path": str(rel),
            "size_bytes": path.stat().st_size,
            "sha256": "",
            "duration_sec": "",
            "sample_rate": "",
            "channels": "",
            "status": "VALID",
            "error": ""
        }
        try:
            row["sha256"] = sha256(path)
            info = sf.info(str(path))
            row["duration_sec"] = round(float(info.duration), 3)
            row["sample_rate"] = int(info.samplerate)
            row["channels"] = int(info.channels)
        except Exception as exc:
            row["status"] = "INVALID"
            row["error"] = str(exc)
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
            "source", "genre", "path", "size_bytes", "sha256", "duration_sec",
            "sample_rate", "channels", "status", "error"
        ])
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["status"] == "VALID"]
    by_genre = {}
    for r in valid:
        by_genre.setdefault(f"{r['source']}::{r['genre']}", 0)
        by_genre[f"{r['source']}::{r['genre']}"] += 1

    hashes = {}
    for r in valid:
        hashes.setdefault(r["sha256"], []).append(r["path"])
    duplicate_clusters = [paths for paths in hashes.values() if len(paths) > 1]

    summary = {
        "total_files": len(rows),
        "valid_files": len(valid),
        "invalid_files": len(rows) - len(valid),
        "class_counts": by_genre,
        "duplicate_clusters": duplicate_clusters,
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/expanded_raw"))
    parser.add_argument("--output", type=Path, default=Path("audit_outputs/expanded_source_inventory.csv"))
    args = parser.parse_args()
    audit(args.root, args.output)
