#!/usr/bin/env python3
"""Expand CodedLang review annotations into auditable phonetic fragments."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from build_fragments import fragment_to_ipa, normalize_for_reading, review_reading


TARGET_CLASSES = {
    "ambiguous_homophone": "歧义同音词",
    "non_lexical_homophone": "非词汇同音词",
    "phonetic": "语音替代",
    "cross-lingual": "跨语言语音编码",
}


def load_dictionary(path):
    entries = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entries[row["code_span"].strip().casefold()].append(row)
    return entries


def split_spans(value):
    return [span.strip() for span in re.split(r"\s*[,，]\s*", value) if span.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = load_dictionary(args.dictionary)
    fragments, review = [], []
    with args.reviews.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            review_classes = {item.strip() for item in row["coded_lang_class"].split(",")}
            selected_classes = review_classes & TARGET_CLASSES.keys()
            if not selected_classes:
                continue
            for number, code_span in enumerate(split_spans(row["code_span"]), 1):
                matches = dictionary.get(code_span.casefold(), [])
                target_matches = [item for item in matches if item["coded_lang_class"] in TARGET_CLASSES]
                unique_decodes = {item["decode"] for item in target_matches}
                issues = []
                if len(unique_decodes) == 1:
                    dictionary_entry = target_matches[0]
                    canonical = dictionary_entry["decode"]
                    extraction_method = "official_dictionary_casefold"
                    category = TARGET_CLASSES[dictionary_entry["coded_lang_class"]]
                elif len(unique_decodes) > 1:
                    dictionary_entry = None
                    canonical = ""
                    extraction_method = "ambiguous_official_dictionary"
                    category = " / ".join(sorted(TARGET_CLASSES[item["coded_lang_class"]] for item in target_matches))
                    issues.append({"type": "ambiguous_dictionary_decode", "candidates": sorted(unique_decodes)})
                else:
                    dictionary_entry = None
                    canonical = ""
                    extraction_method = "not_in_official_dictionary"
                    category = " / ".join(sorted(TARGET_CLASSES[item] for item in selected_classes))
                    issues.append({"type": "missing_dictionary_decode"})

                phonetic_text, normalization_rules = normalize_for_reading(code_span)
                ipa_candidate, ipa_issues = fragment_to_ipa(phonetic_text)
                issues.extend({"type": f"source_{issue['type']}"} for issue in ipa_issues)
                canonical_phonetic_text = ""
                canonical_pinyin_reading = ""
                canonical_ipa_candidate = None
                canonical_ipa_issues = []
                if canonical:
                    canonical_phonetic_text, _ = normalize_for_reading(canonical)
                    canonical_pinyin_reading = review_reading(canonical_phonetic_text)
                    canonical_ipa_candidate, canonical_ipa_issues = fragment_to_ipa(canonical_phonetic_text)
                    issues.extend({"type": f"canonical_{issue['type']}"} for issue in canonical_ipa_issues)

                record = {
                    "fragment_id": f"codedlang:{row['id']}:{number:02d}",
                    "source_dataset": "Crowd-AI-Lab/CodedLang",
                    "review_id": row["id"],
                    "variant_fragment_raw": code_span,
                    "variant_fragment_normalized": code_span,
                    "phonetic_text": phonetic_text,
                    "pinyin_reading": review_reading(phonetic_text),
                    "canonical_fragment": canonical,
                    "canonical_phonetic_text": canonical_phonetic_text,
                    "canonical_pinyin_reading": canonical_pinyin_reading,
                    "ipa": ipa_candidate if not ipa_issues else None,
                    "ipa_candidate": ipa_candidate or None,
                    "canonical_ipa": canonical_ipa_candidate if canonical and not canonical_ipa_issues else None,
                    "canonical_ipa_candidate": canonical_ipa_candidate,
                    "needs_manual_review": bool(issues),
                    "manual_review_reasons": issues,
                    "extraction_method": extraction_method,
                    "category": category,
                    "review_categories": ", ".join(sorted(review_classes)),
                    "official_code_pinyin": dictionary_entry["code_pinyin"] if dictionary_entry else "",
                    "official_decode_pinyin": dictionary_entry["decode_pinyin"] if dictionary_entry else "",
                    "official_code_ipa": dictionary_entry["code_ipa"] if dictionary_entry else "",
                    "official_decode_ipa": dictionary_entry["decode_ipa"] if dictionary_entry else "",
                    "original_review": row["original_review"],
                    "decode_review": row["decode_review"],
                }
                fragments.append(record)
                if issues:
                    review.append(record)

    for name, rows in (("fragments.jsonl", fragments), ("manual_review.jsonl", review)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = list(fragments[0])
    with (args.output_dir / "fragments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in fragments:
            output_row = row.copy()
            output_row["needs_manual_review"] = "是" if row["needs_manual_review"] else "否"
            output_row["manual_review_reasons"] = json.dumps(row["manual_review_reasons"], ensure_ascii=False)
            writer.writerow(output_row)
    print(json.dumps({"fragments": len(fragments), "needs_manual_review": len(review)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
