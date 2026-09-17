#!/usr/bin/env python3
"""Measure actual mT5 token lengths and truncation risk for IPA-to-text."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[IPA]"]
REQUIRED_FIELDS = ("ipa", "canonical_text", "sample_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--source-limits", type=int, nargs="+", default=[512, 768, 1024])
    parser.add_argument("--target-limits", type=int, nargs="+", default=[256, 384, 512])
    parser.add_argument("--selected-source-length", type=int, default=None)
    parser.add_argument("--selected-target-length", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No token lengths were measured")
    return {
        "min": ordered[0],
        "p50": ordered[int((len(ordered) - 1) * 0.50)],
        "p90": ordered[int((len(ordered) - 1) * 0.90)],
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
        "p99": ordered[int((len(ordered) - 1) * 0.99)],
        "max": ordered[-1],
    }


def truncation_summary(lengths: list[int], limits: list[int]) -> dict[str, dict[str, float | int]]:
    total = len(lengths)
    return {
        str(limit): {
            "records_exceeding_limit": sum(length > limit for length in lengths),
            "fraction_exceeding_limit": sum(length > limit for length in lengths) / total,
        }
        for limit in sorted(set(limits))
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install the v3 mT5 training requirements before running this audit.") from error

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array")
    records = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} is not an object")
        missing = [field for field in REQUIRED_FIELDS if not isinstance(row.get(field), str) or not row[field].strip()]
        if missing:
            raise ValueError(f"Record {index} has missing or empty fields: {', '.join(missing)}")
        records.append(row)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    source_lengths, target_lengths = [], []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        sources = [f"[IPA] {row['ipa'].strip()}" for row in batch]
        targets = [row["canonical_text"].strip() for row in batch]
        source_ids = tokenizer(sources, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        target_ids = tokenizer(text_target=targets, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        source_lengths.extend(len(ids) for ids in source_ids)
        target_lengths.extend(len(ids) for ids in target_ids)

    summary = {
        "schema_version": "v3-ipa-token-length-audit-v1",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "model_name": args.model_name,
        "records": len(records),
        "sample_type_counts": dict(sorted(Counter(row["sample_type"] for row in records).items())),
        "input_schema": "[IPA] <segmented IPA>",
        "tokenizer": {
            "vocab_size_after_special_tokens": len(tokenizer),
            "special_tokens_added": SPECIAL_TOKENS,
        },
        "source": {
            "token_length": quantiles(source_lengths),
            "truncation_risk": truncation_summary(source_lengths, args.source_limits),
        },
        "target": {
            "token_length": quantiles(target_lengths),
            "truncation_risk": truncation_summary(target_lengths, args.target_limits),
        },
        "planned_training_limits": {
            "source": args.selected_source_length,
            "target": args.selected_target_length,
        },
        "note": "Lengths include tokenizer special tokens and are measured without truncation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
