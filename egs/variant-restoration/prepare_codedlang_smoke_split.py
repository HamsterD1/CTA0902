#!/usr/bin/env python3
"""Create a leakage-free CodedLang-only unseen-variant smoke-test split."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_pair = defaultdict(list)
    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["extraction_method"] != "official_dictionary_casefold":
                continue
            by_pair[(row["variant_fragment_normalized"], row["canonical_fragment"])].append(row)

    records = []
    for pair_rows in by_pair.values():
        record = pair_rows[0].copy()
        record["source_instance_count"] = len(pair_rows)
        record["source_review_ids"] = sorted({row["review_id"] for row in pair_rows}, key=int)
        record["deduplication_key"] = f"{record['variant_fragment_normalized']} -> {record['canonical_fragment']}"
        records.append(record)

    by_canonical = defaultdict(list)
    for record in records:
        by_canonical[record["canonical_fragment"]].append(record)
    rng = random.Random(args.seed)
    splits = {"train": [], "validation": [], "test_seen_variant": []}
    for canonical in sorted(by_canonical):
        variants = sorted(by_canonical[canonical], key=lambda row: row["variant_fragment_normalized"])
        rng.shuffle(variants)
        if len(variants) >= 3:
            splits["validation"].append(variants.pop())
            splits["test_seen_variant"].append(variants.pop())
        elif len(variants) == 2:
            splits["test_seen_variant"].append(variants.pop())
        splits["train"].extend(variants)

    for name, rows in splits.items():
        for row in rows:
            row["split"] = name
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    fields = list(records[0])
    with (args.output_dir / "all_unique_fragments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in ("train", "validation", "test_seen_variant"):
            for row in splits[name]:
                csv_row = row.copy()
                csv_row["source_review_ids"] = json.dumps(row["source_review_ids"], ensure_ascii=False)
                writer.writerow(csv_row)
    manifest = {
        "source": str(args.source),
        "seed": args.seed,
        "raw_unique_dictionary_records": sum(len(rows) for rows in by_pair.values()),
        "unique_variant_canonical_pairs": len(records),
        "canonical_count": len(by_canonical),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "test_definition": "Seen Canonical + Unseen Variant; no exact variant-canonical pair occurs in train and test.",
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
