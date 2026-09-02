#!/usr/bin/env python3
"""Validate reviewed variant data, split it, and materialize ablation inputs."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REQUIRED_FIELDS = ("id", "variant_text", "ipa", "canonical_text", "variant_type")
CONDITIONS = ("text", "ipa", "text_ipa")


def read_examples(source: Path):
    examples, ids, variants = [], set(), {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            missing = [key for key in REQUIRED_FIELDS if not isinstance(item.get(key), str) or not item[key].strip()]
            if missing:
                raise ValueError(f"{source}:{line_number}: missing or empty {', '.join(missing)}")
            if item["id"] in ids:
                raise ValueError(f"{source}:{line_number}: duplicate id {item['id']!r}")
            ids.add(item["id"])
            variant = item["variant_text"].strip()
            label = item["canonical_text"].strip()
            if variant in variants and variants[variant] != label:
                raise ValueError(f"{source}:{line_number}: variant {variant!r} has conflicting labels")
            variants[variant] = label
            examples.append({key: item[key].strip() for key in REQUIRED_FIELDS})
    if not examples:
        raise ValueError("source contains no examples")
    return examples


def input_for(item, condition):
    if condition == "text":
        return f"[VARIANT] {item['variant_text']}"
    if condition == "ipa":
        return f"[IPA] {item['ipa']}"
    return f"[VARIANT] {item['variant_text']} [IPA] {item['ipa']}"


def assign_splits(examples, split_mode, seed, train_ratio, validation_ratio):
    groups = defaultdict(list)
    for item in examples:
        key = item["canonical_text"] if split_mode == "canonical" else item["variant_text"]
        groups[key].append(item)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    train_end = round(len(keys) * train_ratio)
    validation_end = train_end + round(len(keys) * validation_ratio)
    partitions = {"train": keys[:train_end], "validation": keys[train_end:validation_end], "test": keys[validation_end:]}
    if not all(partitions.values()):
        raise ValueError("not enough distinct split units; need at least three")
    return {name: [item for key in keys_for_split for item in groups[key]] for name, keys_for_split in partitions.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-mode", choices=("canonical", "variant"), default="canonical")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.validation_ratio < 1 or args.train_ratio + args.validation_ratio >= 1:
        parser.error("ratios must be positive and sum to less than one")

    splits = assign_splits(read_examples(args.source), args.split_mode, args.seed, args.train_ratio, args.validation_ratio)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.source), "seed": args.seed, "split_mode": args.split_mode, "counts": {}}
    for split, items in splits.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                item["inputs"] = {condition: input_for(item, condition) for condition in CONDITIONS}
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["counts"][split] = len(items)
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
