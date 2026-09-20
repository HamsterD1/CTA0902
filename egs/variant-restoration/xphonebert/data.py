"""Prompt encoding and metrics shared by the Qwen restoration conditions."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path


def load_descriptor(path: Path) -> dict:
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_path_or_repo", "immutable_revision", "prompt_contract", "text_hidden_size"}
    missing = required - descriptor.keys()
    if missing:
        raise ValueError(f"Descriptor is missing {sorted(missing)}")
    return descriptor


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def user_content(descriptor: dict, row: dict, ipa_section: str = "") -> str:
    return descriptor["prompt_contract"]["user_template"].format(
        variant_text=row["variant_text"], ipa_section=ipa_section
    )


def messages(descriptor: dict, row: dict, ipa_section: str = "") -> list[dict]:
    result = []
    system = descriptor["prompt_contract"]["system_prompt"]
    if system:
        result.append({"role": "system", "content": system})
    result.append({"role": "user", "content": user_content(descriptor, row, ipa_section)})
    return result


def token_ids(encoded) -> list[int]:
    """Extract one unbatched token-id sequence from tokenizer output."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if not isinstance(encoded, list) or any(not isinstance(token, int) for token in encoded):
        raise TypeError(f"Expected one list of token ids, got {type(encoded).__name__}")
    return encoded


def prompt_ids(tokenizer, descriptor: dict, row: dict, ipa_section: str = "") -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages(descriptor, row, ipa_section), tokenize=True, add_generation_prompt=True
    )
    return token_ids(encoded)


def encode_supervised(tokenizer, descriptor: dict, row: dict, ipa_section: str = "") -> dict:
    prompt = prompt_ids(tokenizer, descriptor, row, ipa_section)
    target = token_ids(tokenizer(row["canonical_text"], add_special_tokens=False))
    if tokenizer.eos_token_id is not None and (not target or target[-1] != tokenizer.eos_token_id):
        target.append(tokenizer.eos_token_id)
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target, "prompt_length": len(prompt)}


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, character in enumerate(reference, 1):
        current = [index]
        for other_index, predicted in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (character != predicted)))
        previous = current
    return previous[-1]


def metric_report(records: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = {"all": records}
    for name in ("phonetic", "identity"):
        buckets[name] = [row for row in records if row["sample_type"] == name]
    buckets["multilingual"] = [row for row in records if row.get("language_tags")]
    report = {}
    for name, rows in buckets.items():
        if not rows:
            continue
        exact = sum(row["predictions"][0] == row["canonical_text"] for row in rows)
        top3 = sum(row["canonical_text"] in row["predictions"] for row in rows)
        norm_exact = sum(normalized(row["predictions"][0]) == normalized(row["canonical_text"]) for row in rows)
        chars = sum(len(row["canonical_text"]) for row in rows)
        edits = sum(edit_distance(row["canonical_text"], row["predictions"][0]) for row in rows)
        report[name] = {
            "count": len(rows),
            "top1_exact_match": exact / len(rows),
            "top3_recall": top3 / len(rows),
            "normalized_top1_exact_match": norm_exact / len(rows),
            "character_accuracy": max(0.0, 1 - edits / chars) if chars else 0.0,
        }
    return report


def decode_continuations(tokenizer, generated, prompt_length: int) -> list[str]:
    """Generate with inputs_embeds may or may not return synthetic prompt ids."""
    token_ids = generated[:, prompt_length:] if generated.shape[1] > prompt_length else generated
    return [value.strip() for value in tokenizer.batch_decode(token_ids, skip_special_tokens=True)]
