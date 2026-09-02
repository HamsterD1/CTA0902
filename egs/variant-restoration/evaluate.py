#!/usr/bin/env python3
"""Evaluate decoded JSONL predictions for variant restoration."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, predicted in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != predicted)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    reference_items = [json.loads(line) for line in args.references.read_text(encoding="utf-8").splitlines() if line.strip()]
    prediction_items = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    references = {item["id"]: item for item in reference_items}
    predictions = {item["id"]: item["prediction"].strip() for item in prediction_items}
    missing = set(references) - set(predictions)
    unexpected = set(predictions) - set(references)
    if missing or unexpected:
        raise ValueError(f"prediction ids mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    buckets = defaultdict(list)
    for uid, item in references.items():
        buckets["all"].append((item["canonical_text"], predictions[uid]))
        buckets[item["variant_type"]].append((item["canonical_text"], predictions[uid]))
    report = {}
    for name, pairs in buckets.items():
        total_chars = sum(len(reference) for reference, _ in pairs)
        edits = sum(distance(reference, prediction) for reference, prediction in pairs)
        exact = sum(reference == prediction for reference, prediction in pairs)
        report[name] = {"count": len(pairs), "exact_match_accuracy": exact / len(pairs), "character_accuracy": max(0.0, 1 - edits / total_chars) if total_chars else 0, "cer": edits / total_chars if total_chars else 0}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
