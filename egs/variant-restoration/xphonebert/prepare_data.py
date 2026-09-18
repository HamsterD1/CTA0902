#!/usr/bin/env python3
"""Create immutable, canonical-disjoint XPhoneBERT experiment splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED = ("variant_text", "ipa", "canonical_text", "sample_type")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_key(row: dict[str, str]) -> str:
    return "\0".join(row[field] for field in REQUIRED)


def choose_groups(groups: dict[str, list[dict]], target: int, seed: int) -> set[str]:
    names = list(groups)
    random.Random(seed).shuffle(names)
    chosen: set[str] = set()
    total = 0
    for name in names:
        size = len(groups[name])
        if total == 0 or abs(total + size - target) <= abs(total - target):
            chosen.add(name)
            total += size
    return chosen


def load_rows(training_path: Path, coverage_path: Path | None) -> tuple[list[dict], int]:
    raw_rows = json.loads(training_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("Training export must be a JSON array")
    languages_by_index: dict[int, set[str]] = defaultdict(set)
    if coverage_path:
        for patch in json.loads(coverage_path.read_text(encoding="utf-8")).get("patches", []):
            languages_by_index[patch["record_index"]].update(patch.get("languages", []))
    unique: dict[str, dict] = {}
    duplicates = 0
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Record {index} is not an object")
        row = {field: raw.get(field, "").strip() if isinstance(raw.get(field), str) else "" for field in REQUIRED}
        missing = [field for field, value in row.items() if not value]
        if missing:
            raise ValueError(f"Record {index} has missing fields: {', '.join(missing)}")
        if row["sample_type"] not in {"phonetic", "identity"}:
            raise ValueError(f"Record {index} has invalid sample_type")
        row["language_tags"] = sorted(languages_by_index[index])
        key = record_key(row)
        if key in unique:
            unique[key]["language_tags"] = sorted(set(unique[key]["language_tags"]) | set(row["language_tags"]))
            duplicates += 1
        else:
            unique[key] = row
    rows = list(unique.values())
    for row in rows:
        row["id"] = f"xpb:{sha256_text(record_key(row))[:20]}"
    return rows, duplicates


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, duplicates = load_rows(args.input, args.coverage)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["canonical_text"]].append(row)
    test_groups = choose_groups(groups, round(len(rows) * 0.10), args.seed)
    remaining = {name: values for name, values in groups.items() if name not in test_groups}
    validation_groups = choose_groups(remaining, round(len(rows) * 0.10), args.seed + 1)
    splits = {"train": [], "validation": [], "test": []}
    for canonical, values in groups.items():
        name = "test" if canonical in test_groups else "validation" if canonical in validation_groups else "train"
        splits[name].extend(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", values)
    manifest = {
        "version": "xphonebert-v1",
        "input": str(args.input),
        "input_sha256": sha256_text(args.input.read_text(encoding="utf-8")),
        "seed": args.seed,
        "raw_records": len(rows) + duplicates,
        "unique_records": len(rows),
        "exact_duplicate_records_removed": duplicates,
        "split_policy": "canonical_text_grouped_80_10_10",
        "split_definition": "Each canonical_text belongs to exactly one split.",
        "splits": {
            name: {
                "records": len(values),
                "canonical_targets": len({row["canonical_text"] for row in values}),
                "sample_type_counts": dict(sorted(Counter(row["sample_type"] for row in values).items())),
                "multilingual_records": sum(bool(row["language_tags"]) for row in values),
            }
            for name, values in splits.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
