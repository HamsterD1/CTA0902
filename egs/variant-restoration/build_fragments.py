#!/usr/bin/env python3
"""Build fragment-level restoration records and conservative IPA candidates."""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from pypinyin import Style, pinyin


PINYIN_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")
INITIAL_IPA = {"b": "p", "p": "pʰ", "m": "m", "f": "f", "d": "t", "t": "tʰ", "n": "n", "l": "l", "g": "k", "k": "kʰ", "h": "x", "j": "tɕ", "q": "tɕʰ", "x": "ɕ", "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ɻ", "z": "ts", "c": "tsʰ", "s": "s", "y": "j", "w": "w"}
FINAL_IPA = {"a": "a", "o": "o", "e": "ɤ", "ai": "ai", "ei": "ei", "ao": "ɑʊ", "ou": "oʊ", "an": "an", "en": "ən", "ang": "ɑŋ", "eng": "ɤŋ", "ong": "ʊŋ", "i": "i", "ia": "ia", "ie": "iɛ", "iao": "iɑʊ", "io": "iɔ", "iu": "iou", "iou": "iou", "ian": "iɛn", "in": "in", "iang": "iɑŋ", "ing": "iŋ", "iong": "iʊŋ", "u": "u", "ua": "ua", "uo": "uɔ", "uai": "uai", "ui": "uei", "uan": "uan", "un": "uən", "uang": "uɑŋ", "ueng": "uəŋ", "ü": "y", "üe": "yɛ", "üan": "yɛn", "ün": "yn", "er": "əɻ", "ê": "ɛ"}
SYMBOL_READINGS = {"🐔": "鸡", "🐟": "鱼", "🌨️": "雪", "+": "加"}
DIGIT_READINGS = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
SPECIAL_LATIN_IPA = {"vb": "viː puɔ"}
RETAINED_LATIN_FORMS = {"neinei"}


def split_mapping(value):
    value = value.strip().replace("→", "->")
    if value == "无":
        return []
    pairs = []
    for segment in value.split("；"):
        parts = re.split(r"\s*->\s*", segment.strip(), maxsplit=1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid mapping segment: {segment!r}")
        pairs.append(tuple(parts))
    return pairs


def pinyin_syllable_to_ipa(syllable):
    match = re.fullmatch(r"([a-züê]+)([1-5]?)", syllable.lower())
    if not match:
        return None, "invalid_pinyin"
    body, tone = match.groups()
    body = body.replace("v", "ü")
    if body in {"zhi", "chi", "shi", "ri"}:
        return INITIAL_IPA[body[:-1]] + "ɻ̩", None
    if body in {"zi", "ci", "si"}:
        return INITIAL_IPA[body[:-1]] + "ɹ̩", None
    for initial in PINYIN_INITIALS:
        if body.startswith(initial):
            final = body[len(initial):]
            break
    else:
        initial, final = "", body
    # Pinyin writes /y/ as u after j/q/x and as yu after a zero initial.
    if initial in {"j", "q", "x"} and final.startswith("u"):
        final = "ü" + final[1:]
    # In pinyin, bare "o" after labials is the /uɔ/ final (bo, po, mo, fo).
    if initial in {"b", "p", "m", "f"} and final == "o":
        final = "uo"
    if initial == "y":
        initial = ""
        final = {"yi": "i", "yin": "in", "ying": "ing", "ya": "ia", "yao": "iao", "ye": "ie", "you": "iou", "yan": "ian", "yang": "iang", "yong": "iong", "yo": "io", "yu": "ü", "yue": "üe", "yuan": "üan", "yun": "ün"}.get(body, body[1:])
    if initial == "w":
        initial = ""
        final = {"wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "ui", "wan": "uan", "wang": "uang", "wen": "un", "weng": "ueng"}.get(body, body[1:])
    if final not in FINAL_IPA:
        return None, "unsupported_pinyin_final"
    return INITIAL_IPA.get(initial, "") + FINAL_IPA[final], None


def latin_span_to_ipa(span):
    """Return a conservative IPA candidate for a contiguous Latin-word span.

    A fully valid pinyin span is treated as Mandarin. Every other span is read
    as British English by espeak-ng. Exceptions such as ``neinei`` are retained
    explicitly above rather than guessed as English.
    """
    normalized_span = span.lower()
    if normalized_span in RETAINED_LATIN_FORMS:
        return None, [{"type": "retained_latin_form", "text": span}]
    if normalized_span in SPECIAL_LATIN_IPA:
        return SPECIAL_LATIN_IPA[normalized_span], []
    tokens = re.findall(r"[A-Za-züê]+", normalized_span)
    pinyin_results = [pinyin_syllable_to_ipa(token) for token in tokens]
    if tokens and all(ipa is not None for ipa, _ in pinyin_results):
        ipa = " ".join(value for value, _ in pinyin_results)
        issues = [{"type": issue, "text": token} for token, (_, issue) in zip(tokens, pinyin_results) if issue]
        return ipa, issues
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "en-gb", "--ipa=3", span],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, [{"type": "unresolved_latin_span", "text": span}]
    ipa = result.stdout.strip().replace("\u200d", "")
    if not ipa:
        return None, [{"type": "unresolved_latin_span", "text": span}]
    return ipa, []


def normalize_for_reading(text):
    """Expand confirmed non-text forms before the shared Chinese G2P step."""
    output, rules = [], []
    index = 0
    while index < len(text):
        if text.startswith("4000+", index):
            output.append("四千加")
            rules.append({"surface": "4000+", "text": "四千加"})
            index += 5
            continue
        digit_letter = re.match(r"4\s*i", text[index:], flags=re.IGNORECASE)
        if digit_letter:
            output.append("四 ai")
            rules.append({"surface": digit_letter.group(0), "text": "四 ai"})
            index += len(digit_letter.group(0))
            continue
        char = text[index]
        matched_symbol = next((symbol for symbol in SYMBOL_READINGS if text.startswith(symbol, index)), None)
        if char in DIGIT_READINGS:
            output.append(DIGIT_READINGS[char])
            rules.append({"surface": char, "text": DIGIT_READINGS[char]})
        elif matched_symbol:
            output.append(SYMBOL_READINGS[matched_symbol])
            rules.append({"surface": matched_symbol, "text": SYMBOL_READINGS[matched_symbol]})
            index += len(matched_symbol) - 1
        else:
            output.append(char)
        index += 1
    return "".join(output), rules


def fragment_to_ipa(text):
    output, issues = [], []
    index = 0
    while index < len(text):
        char = text[index]
        if "\u4e00" <= char <= "\u9fff":
            match = re.match(r"[\u4e00-\u9fff]+", text[index:])
            hanzi_span = match.group(0)
            # Phrase-level G2P disambiguates context such as 策划/划船 and
            # 几把/几乎 without treating dictionary-only readings as errors.
            readings = pinyin(hanzi_span, style=Style.TONE3, strict=False)
            for hanzi, reading in zip(hanzi_span, readings):
                ipa, issue = pinyin_syllable_to_ipa(reading[0])
                if ipa is None:
                    issues.append({"type": issue, "text": hanzi})
                else:
                    output.append(ipa)
            index += len(hanzi_span)
            continue
        if char.isascii() and char.isalpha():
            match = re.match(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", text[index:])
            span = match.group(0)
            ipa, token_issues = latin_span_to_ipa(span)
            if ipa is None:
                issues.extend(token_issues)
            else:
                output.append(ipa)
                issues.extend(token_issues)
            index += len(span)
            continue
        if char.isspace() or char in "，。！？；：、,.!?()（）[]【】_-":
            index += 1
            continue
        issues.append({"type": "unresolved_symbol_or_digit", "text": char})
        index += 1
    return " ".join(output), issues


def review_reading(text):
    """Render the pre-IPA pronunciation in human-readable pinyin/English."""
    output = []
    index = 0
    while index < len(text):
        char = text[index]
        if "\u4e00" <= char <= "\u9fff":
            match = re.match(r"[\u4e00-\u9fff]+", text[index:])
            hanzi_span = match.group(0)
            output.append(" ".join(item[0] for item in pinyin(hanzi_span, style=Style.NORMAL, strict=False)))
            index += len(hanzi_span)
            continue
        if char.isascii() and char.isalpha():
            match = re.match(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", text[index:])
            span = match.group(0)
            output.append("V bo" if span.lower() == "vb" else span)
            index += len(span)
            continue
        output.append(char)
        index += 1
    return " ".join(part for part in output if part.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    excluded, fragments, review = [], [], []
    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["macro_category"].strip() != "拟音":
                excluded.append({"id": row["id"], "reason": "macro_category_outside_mvp_scope", "macro_category": row["macro_category"]})
                continue
            normalized = row["语义片段映射（字母归一化）"]
            try:
                normalized_pairs = split_mapping(normalized)
                raw_pairs = split_mapping(row["语义片段映射"])
            except ValueError as error:
                excluded.append({"id": row["id"], "reason": str(error), "mapping": normalized})
                continue
            if not normalized_pairs:
                excluded.append({"id": row["id"], "reason": "no_semantic_fragment_mapping", "mapping": normalized})
                continue
            if len(normalized_pairs) != len(raw_pairs):
                excluded.append({"id": row["id"], "reason": "raw_and_normalized_mapping_count_mismatch", "mapping": normalized})
                continue
            for number, ((raw_source, raw_target), (source, target)) in enumerate(zip(raw_pairs, normalized_pairs), 1):
                phonetic_text, normalization_rules = normalize_for_reading(source)
                ipa_candidate, issues = fragment_to_ipa(phonetic_text)
                canonical_phonetic_text, canonical_normalization_rules = normalize_for_reading(target)
                canonical_ipa_candidate, canonical_issues = fragment_to_ipa(canonical_phonetic_text)
                record = {
                    "fragment_id": f"{row['id']}:{number:02d}", "source_row_id": row["id"],
                    "variant_fragment_raw": raw_source, "variant_fragment_normalized": source,
                    "phonetic_text": phonetic_text, "pinyin_reading": review_reading(phonetic_text),
                    "normalization_rules": normalization_rules,
                    "canonical_fragment": target, "ipa_candidate": ipa_candidate or None,
                    "ipa": ipa_candidate if not issues else None, "ipa_status": "needs_review" if issues else "auto_ready",
                    "ipa_issues": issues, "original_variant_text": row["原始变种文本"],
                    "canonical_phonetic_text": canonical_phonetic_text,
                    "canonical_pinyin_reading": review_reading(canonical_phonetic_text),
                    "canonical_normalization_rules": canonical_normalization_rules,
                    "canonical_ipa_candidate": canonical_ipa_candidate or None,
                    "canonical_ipa": canonical_ipa_candidate if not canonical_issues else None,
                    "canonical_ipa_status": "needs_review" if canonical_issues else "auto_ready",
                    "canonical_ipa_issues": canonical_issues,
                    "original_canonical_text": row["还原答案"], "macro_category": row["macro_category"],
                    "category": row["category"], "letter_normalization_applied": raw_source != source,
                }
                fragments.append(record)
                if issues:
                    review.append(record)
    for name, records in (("fragments.jsonl", fragments), ("ipa_review.jsonl", review), ("excluded.jsonl", excluded)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_fields = [
        "fragment_id", "source_row_id", "variant_fragment_raw", "variant_fragment_normalized",
        "phonetic_text", "pinyin_reading", "canonical_fragment", "canonical_phonetic_text",
        "canonical_pinyin_reading", "canonical_ipa", "canonical_ipa_candidate",
        "canonical_needs_manual_review", "canonical_manual_review_reasons", "ipa", "ipa_candidate", "needs_manual_review",
        "manual_review_reasons", "normalization_rules", "macro_category", "category",
        "original_variant_text", "original_canonical_text",
    ]
    with (args.output_dir / "fragments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in fragments:
            writer.writerow({
                "fragment_id": record["fragment_id"],
                "source_row_id": record["source_row_id"],
                "variant_fragment_raw": record["variant_fragment_raw"],
                "variant_fragment_normalized": record["variant_fragment_normalized"],
                "phonetic_text": record["phonetic_text"],
                "pinyin_reading": record["pinyin_reading"],
                "canonical_fragment": record["canonical_fragment"],
                "canonical_phonetic_text": record["canonical_phonetic_text"],
                "canonical_pinyin_reading": record["canonical_pinyin_reading"],
                "canonical_ipa": record["canonical_ipa"] or "",
                "canonical_ipa_candidate": record["canonical_ipa_candidate"] or "",
                "canonical_needs_manual_review": "是" if record["canonical_ipa_status"] == "needs_review" else "否",
                "canonical_manual_review_reasons": "; ".join(issue["type"] for issue in record["canonical_ipa_issues"]),
                "ipa": record["ipa"] or "",
                "ipa_candidate": record["ipa_candidate"] or "",
                "needs_manual_review": "是" if record["ipa_status"] == "needs_review" else "否",
                "manual_review_reasons": "; ".join(issue["type"] for issue in record["ipa_issues"]),
                "normalization_rules": json.dumps(record["normalization_rules"], ensure_ascii=False),
                "macro_category": record["macro_category"],
                "category": record["category"],
                "original_variant_text": record["original_variant_text"],
                "original_canonical_text": record["original_canonical_text"],
            })
    print(json.dumps({"fragments": len(fragments), "auto_ready": sum(not x["ipa_issues"] for x in fragments), "needs_review": len(review), "excluded": len(excluded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
