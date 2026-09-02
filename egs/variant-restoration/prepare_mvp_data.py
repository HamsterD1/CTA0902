#!/usr/bin/env python3
"""Merge confirmed local and CodedLang phonetic fragments into an MVP split."""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


LOCAL_CATEGORY_MAP = {
    "歧义同音词": "歧义同音词",
    "非词汇同音词": "非词汇同音词",
    "语音符号替代": "语音替代",
    "跨语言语音编码（含方言，可称非普通话音近）": "跨语言语音编码",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def input_for(record, condition):
    if condition == "text":
        return f"[VARIANT] {record['variant_text']}"
    if condition == "ipa":
        return f"[IPA] {record['ipa']}"
    return f"[VARIANT] {record['variant_text']} [IPA] {record['ipa']}"


def load_local(path):
    records = []
    for row in read_csv(path):
        if row["needs_manual_review"] != "否" or row["canonical_needs_manual_review"] != "否":
            continue
        category = LOCAL_CATEGORY_MAP.get(row["category"])
        if category is None:
            continue
        records.append({
            "id": f"local:{row['fragment_id']}",
            "source_dataset": "local_657_homophone_fragments",
            "source_record_id": row["fragment_id"],
            "variant_text": row["variant_fragment_normalized"],
            "variant_pinyin": row["pinyin_reading"],
            "ipa": row["ipa"],
            "ipa_source": "generated_no_tone",
            "canonical_text": row["canonical_fragment"],
            "canonical_pinyin": row["canonical_pinyin_reading"],
            "canonical_ipa": row["canonical_ipa"],
            "variant_type": category,
        })
    return records


def load_codedlang(path):
    records = []
    for row in read_csv(path):
        if row["extraction_method"] != "official_dictionary_casefold":
            continue
        ipa = row["ipa"] or row["official_code_ipa"] or row["ipa_candidate"]
        canonical_ipa = row["canonical_ipa"] or row["official_decode_ipa"] or row["canonical_ipa_candidate"]
        if not ipa or not canonical_ipa:
            raise ValueError(f"CodedLang confirmed record lacks IPA: {row['fragment_id']}")
        records.append({
            "id": row["fragment_id"],
            "source_dataset": "Crowd-AI-Lab/CodedLang",
            "source_record_id": row["review_id"],
            "variant_text": row["variant_fragment_normalized"],
            "variant_pinyin": row["pinyin_reading"] or row["official_code_pinyin"],
            "ipa": ipa,
            "ipa_source": "generated_no_tone" if row["ipa"] else "official_codedlang",
            "canonical_text": row["canonical_fragment"],
            "canonical_pinyin": row["canonical_pinyin_reading"] or row["official_decode_pinyin"],
            "canonical_ipa": canonical_ipa,
            "variant_type": row["category"],
        })
    return records


def choose_validation_groups(groups, test_canonicals, target_size, seed):
    """Hold out complete canonical groups, keeping test canonicals train-visible."""
    candidates = [(canonical, pairs) for canonical, pairs in groups.items() if canonical not in test_canonicals]
    random.Random(seed).shuffle(candidates)
    selected, total = set(), 0
    for canonical, pairs in candidates:
        size = sum(len(records) for _, records in pairs)
        if total == 0 or abs((total + size) - target_size) <= abs(total - target_size):
            selected.add(canonical)
            total += size
    return selected


def make_splits(records, validation_size, seed):
    by_pair = defaultdict(list)
    for record in records:
        by_pair[(record["variant_text"], record["canonical_text"])].append(record)
    by_canonical = defaultdict(list)
    for pair, pair_records in by_pair.items():
        by_canonical[pair[1]].append((pair, pair_records))

    rng = random.Random(seed)
    test_pairs, remaining = set(), defaultdict(list)
    for canonical, pair_groups in by_canonical.items():
        groups = pair_groups.copy()
        rng.shuffle(groups)
        if len(groups) >= 2:
            pair, _ = groups.pop()
            test_pairs.add(pair)
        remaining[canonical] = groups
    test_canonicals = {pair[1] for pair in test_pairs}
    validation_canonicals = choose_validation_groups(remaining, test_canonicals, validation_size, seed + 1)

    splits = {"train": [], "validation": [], "test_seen_canonical_unseen_variant": []}
    for pair, pair_records in by_pair.items():
        canonical = pair[1]
        if pair in test_pairs:
            split = "test_seen_canonical_unseen_variant"
        elif canonical in validation_canonicals:
            split = "validation"
        else:
            split = "train"
        for record in pair_records:
            record = record.copy()
            record["split"] = split
            record["inputs"] = {condition: input_for(record, condition) for condition in ("text", "ipa", "text_ipa")}
            splits[split].append(record)
    return splits, by_pair


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def training_record(record):
    """The minimal, model-facing representation of one (variant, IPA, target) triple."""
    return {
        "id": record["id"],
        "variant_text": record["variant_text"],
        "ipa": record["ipa"],
        "canonical_text": record["canonical_text"],
        "variant_type": record["variant_type"],
        "inputs": record["inputs"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-source", type=Path, required=True)
    parser.add_argument("--codedlang-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local = load_local(args.local_source)
    codedlang = load_codedlang(args.codedlang_source)
    records = local + codedlang
    splits, by_pair = make_splits(records, args.validation_size, args.seed)
    for split, split_records in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_records)
    write_jsonl(args.output_dir / "all_confirmed.jsonl", records)
    training_dir = args.output_dir / "seq2seq-v1"
    training_dir.mkdir(exist_ok=True)
    training_splits = {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test_seen_canonical_unseen_variant"],
    }
    for split, split_records in training_splits.items():
        write_jsonl(training_dir / f"{split}.jsonl", [training_record(record) for record in split_records])

    manifest = {
        "seed": args.seed,
        "selection": {
            "local": "needs_manual_review=否 AND canonical_needs_manual_review=否 AND target phonetic category",
            "codedlang": "extraction_method=official_dictionary_casefold",
        },
        "source_counts": {"local": len(local), "codedlang": len(codedlang), "total": len(records)},
        "unique_variant_canonical_pairs": len(by_pair),
        "split_counts": {split: len(split_records) for split, split_records in splits.items()},
        "split_definitions": {
            "test_seen_canonical_unseen_variant": "The canonical target occurs in training, but the exact variant-target pair does not.",
            "validation": "Canonical targets are absent from training and test; this avoids all pair and canonical leakage.",
        },
        "training_data_dir": "seq2seq-v1",
        "training_schema": "(variant_text, ipa, canonical_text), with inputs for text/ipa/text_ipa ablations.",
        "ipa_note": "Local records use generated no-tone IPA. CodedLang records use generated no-tone IPA where available, otherwise its official IPA annotation.",
        "variant_type_counts": {split: dict(Counter(row["variant_type"] for row in split_records)) for split, split_records in splits.items()},
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
