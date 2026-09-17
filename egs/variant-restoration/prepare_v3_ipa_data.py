#!/usr/bin/env python3
"""Build deterministic, leakage-safe IPA-to-text splits from v3 triples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = ("variant_text", "ipa", "canonical_text", "sample_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--identity-sample-ratio", type=float, default=0.10)
    parser.add_argument("--input-mode", choices=("ipa", "variant_ipa"), default="ipa")
    parser.add_argument("--keep-exact-duplicates", action="store_true")
    parser.add_argument("--use-all-training-records", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_key(record: dict[str, str]) -> str:
    return "\0".join(record[field] for field in REQUIRED_FIELDS)


def load_records(path: Path, keep_exact_duplicates: bool) -> tuple[list[dict[str, str]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    records, unique = [], {}
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Record {index} is not an object")
        missing = [field for field in REQUIRED_FIELDS if not isinstance(raw.get(field), str) or not raw[field].strip()]
        if missing:
            raise ValueError(f"Record {index} has missing or empty fields: {', '.join(missing)}")
        if raw["sample_type"] not in {"phonetic", "identity"}:
            raise ValueError(f"Record {index} has unsupported sample_type={raw['sample_type']!r}")
        record = {field: raw[field].strip() for field in REQUIRED_FIELDS}
        records.append(record)
        unique.setdefault(record_key(record), record)
    duplicates = len(records) - len(unique)
    return (records if keep_exact_duplicates else list(unique.values())), duplicates


def choose_groups(groups: dict[str, list[dict[str, str]]], target_size: int, seed: int) -> set[str]:
    """Choose complete canonical-target groups nearest to a row-count target."""
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    selected: set[str] = set()
    total = 0
    for canonical in ordered:
        group_size = len(groups[canonical])
        if total == 0 or abs(total + group_size - target_size) <= abs(total - target_size):
            selected.add(canonical)
            total += group_size
    return selected


def split_records(records: list[dict[str, str]], validation_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict[str, str]]]:
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation/test ratios must be positive and sum to less than one")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[record["canonical_text"]].append(record)
    test_groups = choose_groups(groups, round(len(records) * test_ratio), seed)
    remaining = {canonical: rows for canonical, rows in groups.items() if canonical not in test_groups}
    validation_groups = choose_groups(remaining, round(len(records) * validation_ratio), seed + 1)
    splits = {"train": [], "validation": [], "test": []}
    for canonical, rows in groups.items():
        split = "test" if canonical in test_groups else "validation" if canonical in validation_groups else "train"
        splits[split].extend(rows)
    if not all(splits.values()):
        raise ValueError("A split is empty; use a larger dataset or different ratios")
    return splits


def sample_training_identities(records: list[dict[str, str]], ratio: float, seed: int) -> list[dict[str, str]]:
    if not 0 <= ratio < 1:
        raise ValueError("identity-sample-ratio must be in [0, 1)")
    phonetic = [record for record in records if record["sample_type"] == "phonetic"]
    identities = [record for record in records if record["sample_type"] == "identity"]
    requested = round(len(phonetic) * ratio / (1 - ratio)) if ratio else 0
    rng = random.Random(seed)
    rng.shuffle(identities)
    sampled = phonetic + identities[:min(requested, len(identities))]
    rng.shuffle(sampled)
    return sampled


def input_text(record: dict[str, str], input_mode: str) -> str:
    if input_mode == "ipa":
        return f"[IPA] {record['ipa']}"
    return f"[VARIANT] {record['variant_text']} [IPA] {record['ipa']}"


def model_record(record: dict[str, str], input_mode: str, duplicate_index: int) -> dict[str, str]:
    identifier = f"v3:{digest(record_key(record))[:20]}"
    if duplicate_index:
        identifier = f"{identifier}:{duplicate_index}"
    return {
        "id": identifier,
        "ipa": record["ipa"],
        "canonical_text": record["canonical_text"],
        "sample_type": record["sample_type"],
        "input": input_text(record, input_mode),
    }


def write_jsonl(path: Path, records: list[dict[str, str]], input_mode: str) -> None:
    seen: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            key = record_key(record)
            duplicate_index = seen[key]
            seen[key] += 1
            handle.write(json.dumps(model_record(record, input_mode, duplicate_index), ensure_ascii=False) + "\n")


def counts(records: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(record["sample_type"] for record in records).items()))


def main() -> None:
    args = parse_args()
    records, exact_duplicates = load_records(args.input, args.keep_exact_duplicates)
    splits = split_records(records, args.validation_ratio, args.test_ratio, args.seed)
    unsampled_train = splits["train"]
    splits["train"] = unsampled_train if args.use_all_training_records else sample_training_identities(
        unsampled_train, args.identity_sample_ratio, args.seed + 2
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows, args.input_mode)
    manifest = {
        "schema_version": "full-data-ipa-mt5-split-v1",
        "input": str(args.input),
        "input_sha256": digest(args.input.read_text(encoding="utf-8")),
        "seed": args.seed,
        "raw_records": len(records) if args.keep_exact_duplicates else len(records) + exact_duplicates,
        "unique_records": len({record_key(record) for record in records}),
        "exact_duplicate_records_found": exact_duplicates,
        "exact_duplicate_records_removed": 0 if args.keep_exact_duplicates else exact_duplicates,
        "exact_duplicate_records_retained": exact_duplicates if args.keep_exact_duplicates else 0,
        "unique_canonical_targets": len({record["canonical_text"] for record in records}),
        "split_policy": "canonical_text_grouped",
        "split_definition": "A canonical_text appears in exactly one split, preventing target and exact-triple leakage.",
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "identity_sample_ratio_requested": args.identity_sample_ratio,
        "input_mode": args.input_mode,
        "use_all_training_records": args.use_all_training_records,
        "train_before_identity_sampling": {"records": len(unsampled_train), "sample_type_counts": counts(unsampled_train)},
        "splits": {name: {"records": len(rows), "sample_type_counts": counts(rows), "canonical_targets": len({row['canonical_text'] for row in rows})} for name, rows in splits.items()},
        "schema": {"input": "[IPA] <segmented IPA>" if args.input_mode == "ipa" else "[VARIANT] <variant_text> [IPA] <segmented IPA>", "target": "canonical_text", "audit_field": "sample_type"},
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
