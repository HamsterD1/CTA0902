#!/usr/bin/env python3
"""Paired, prediction-only diagnosis for frozen Text-SFT and Fusion outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from data import edit_distance, load_jsonl


TEXT_BINS = ((24, "<=24"), (32, "25-32"), (48, "33-48"), (64, "49-64"), (None, ">=65"))
IPA_BINS = ((4, "<=4"), (8, "5-8"), (16, "9-16"), (None, ">=17"))


def gold_rank(row: dict) -> int | None:
    try:
        return row["predictions"].index(row["canonical_text"])
    except ValueError:
        return None


def bin_label(value: int, bins: tuple[tuple[int | None, str], ...]) -> str:
    for maximum, label in bins:
        if maximum is None or value <= maximum:
            return label
    raise AssertionError("unreachable")


def percent(numerator: int, denominator: int) -> float:
    return 100 * numerator / denominator if denominator else 0.0


def metrics(rows: list[dict]) -> dict:
    count = len(rows)
    chars = sum(len(row["canonical_text"]) for row in rows)
    edits = sum(edit_distance(row["canonical_text"], row["predictions"][0]) for row in rows)
    return {
        "count": count,
        "top1_percent": percent(sum(gold_rank(row) == 0 for row in rows), count),
        "top3_percent": percent(sum(gold_rank(row) is not None for row in rows), count),
        "char_acc_percent": percent(chars - edits, chars) if chars else 0.0,
    }


def stratify(base: list[dict], fusion: list[dict], labeler) -> dict:
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for base_row, fusion_row in zip(base, fusion, strict=True):
        groups[labeler(base_row)].append((base_row, fusion_row))
    report = {}
    for label, pairs in sorted(groups.items()):
        base_rows, fusion_rows = map(list, zip(*pairs, strict=True))
        base_metric, fusion_metric = metrics(base_rows), metrics(fusion_rows)
        report[label] = {
            "base": base_metric,
            "fusion": fusion_metric,
            "delta_pp": {key: round(fusion_metric[key] - base_metric[key], 4) for key in ("top1_percent", "top3_percent", "char_acc_percent")},
        }
    return report


def validate(base: list[dict], fusion: list[dict]) -> None:
    if len(base) != len(fusion):
        raise ValueError(f"Prediction counts differ: {len(base)} vs {len(fusion)}")
    for index, (base_row, fusion_row) in enumerate(zip(base, fusion, strict=True)):
        for key in ("id", "canonical_text", "sample_type", "variant_text", "ipa"):
            if base_row.get(key) != fusion_row.get(key):
                raise ValueError(f"{key} mismatch at row {index}")
        if len(base_row["predictions"]) != 3 or len(fusion_row["predictions"]) != 3:
            raise ValueError(f"Expected exactly three beam candidates at row {index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--fusion-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, default=20)
    args = parser.parse_args()
    base, fusion = load_jsonl(args.base_predictions), load_jsonl(args.fusion_predictions)
    validate(base, fusion)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transitions, rank_transitions, candidate_changes = Counter(), Counter(), Counter()
    loss_rank, gain_rank = Counter(), Counter()
    concentration: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    for base_row, fusion_row in zip(base, fusion, strict=True):
        base_rank, fusion_rank = gold_rank(base_row), gold_rank(fusion_row)
        base_top1, fusion_top1 = base_rank == 0, fusion_rank == 0
        base_top3, fusion_top3 = base_rank is not None, fusion_rank is not None
        transitions[f"top1:{'correct' if base_top1 else 'wrong'}->{'correct' if fusion_top1 else 'wrong'}"] += 1
        rank_transitions[f"{base_rank if base_rank is not None else 'miss'}->{fusion_rank if fusion_rank is not None else 'miss'}"] += 1
        same_set = set(base_row["predictions"]) == set(fusion_row["predictions"])
        candidate_changes["same_top3_set" if same_set else "different_top3_set"] += 1
        if base_row["predictions"][0] != fusion_row["predictions"][0]:
            candidate_changes["top1_changed_same_set" if same_set else "top1_changed_different_set"] += 1
        categories = []
        if base_top1 and not fusion_top1:
            categories.append("base_top1_correct_to_fusion_wrong")
            loss_rank[str(fusion_rank if fusion_rank is not None else "missing")] += 1
        if not base_top1 and fusion_top1:
            categories.append("base_top1_wrong_to_fusion_correct")
            gain_rank[str(base_rank if base_rank is not None else "missing")] += 1
        if base_top3 and not fusion_top3:
            categories.append("base_top3_correct_to_fusion_wrong")
        if not base_top3 and fusion_top3:
            categories.append("base_top3_wrong_to_fusion_correct")
        if same_set and base_rank != fusion_rank:
            categories.append("same_candidate_set_rank_changed")
        axes = {
            "sample_type": base_row["sample_type"],
            "language": ",".join(base_row.get("language_tags", [])) or "untagged",
            "text_token_length": bin_label(base_row["text_token_count"], TEXT_BINS),
            "ipa_phoneme_length": bin_label(base_row["ipa_phoneme_count"], IPA_BINS),
        }
        for category in categories:
            for axis, value in axes.items():
                concentration[f"{category}:{axis}"][value] += 1
            if len(examples[category]) < args.examples_per_category:
                examples[category].append({
                    "id": base_row["id"], "sample_type": base_row["sample_type"],
                    "variant_text": base_row["variant_text"], "ipa": base_row["ipa"],
                    "canonical_text": base_row["canonical_text"],
                    "base_predictions": base_row["predictions"], "fusion_predictions": fusion_row["predictions"],
                    "base_gold_rank": base_rank, "fusion_gold_rank": fusion_rank,
                    "text_token_count": base_row["text_token_count"], "ipa_phoneme_count": base_row["ipa_phoneme_count"],
                })
    base_all, fusion_all = metrics(base), metrics(fusion)
    report = {
        "protocol": {"records": len(base), "beam_candidates_per_record": 3, "notes": "Prediction-only paired diagnosis; logits and generation scores were not saved by the evaluator."},
        "overall": {"base": base_all, "fusion": fusion_all, "delta_pp": {key: round(fusion_all[key] - base_all[key], 4) for key in ("top1_percent", "top3_percent", "char_acc_percent")}},
        "by_sample_type": stratify(base, fusion, lambda row: row["sample_type"]),
        "by_language": stratify(base, fusion, lambda row: ",".join(row.get("language_tags", [])) or "untagged"),
        "by_text_token_length": stratify(base, fusion, lambda row: bin_label(row["text_token_count"], TEXT_BINS)),
        "by_ipa_phoneme_length": stratify(base, fusion, lambda row: bin_label(row["ipa_phoneme_count"], IPA_BINS)),
        "top1_transitions": dict(sorted(transitions.items())),
        "gold_rank_transitions": dict(sorted(rank_transitions.items())),
        "candidate_changes": dict(sorted(candidate_changes.items())),
        "base_top1_correct_to_fusion_wrong_fusion_gold_rank": dict(sorted(loss_rank.items())),
        "base_top1_wrong_to_fusion_correct_base_gold_rank": dict(sorted(gain_rank.items())),
        "concentration_counts": {key: dict(sorted(value.items())) for key, value in sorted(concentration.items())},
    }
    (args.output_dir / "paired_diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "representative_examples.jsonl").open("w", encoding="utf-8") as output:
        for category in sorted(examples):
            for row in examples[category]:
                output.write(json.dumps({"category": category, **row}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
