#!/usr/bin/env python3
"""Validate models and data before any XPhoneBERT smoke or training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data import token_ids
from model_contract import context_limit, text_hidden_size


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def round_up(value: int, multiple: int = 128) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def chat_ids(tokenizer, descriptor: dict, variant_text: str, ipa_section: str) -> list[int]:
    contract = descriptor["prompt_contract"]
    user = contract["user_template"].format(variant_text=variant_text, ipa_section=ipa_section)
    messages = []
    if contract["system_prompt"]:
        messages.append({"role": "system", "content": contract["system_prompt"]})
    messages.append({"role": "user", "content": user})
    return token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-descriptor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xphonebert-model", default="vinai/xphonebert-base")
    parser.add_argument("--xphonebert-revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    descriptor = json.loads(args.model_descriptor.read_text(encoding="utf-8"))
    required = {"model_path_or_repo", "immutable_revision", "text_hidden_size", "prompt_contract", "prompt_contract_sha256"}
    missing = required - descriptor.keys()
    if missing:
        raise SystemExit(f"Invalid model descriptor; missing {sorted(missing)}")
    if descriptor["text_hidden_size"] != 4096:
        raise SystemExit("The Qwen descriptor must declare text_hidden_size=4096")
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install the H100 requirements before preflight") from error
    qwen_options = {"revision": descriptor["immutable_revision"], "trust_remote_code": False, "local_files_only": args.local_files_only}
    xpb_options = {"revision": args.xphonebert_revision, "trust_remote_code": False, "local_files_only": args.local_files_only}
    qwen_config = AutoConfig.from_pretrained(descriptor["model_path_or_repo"], **qwen_options)
    qwen_tokenizer = AutoTokenizer.from_pretrained(descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), **qwen_options)
    xpb_config = AutoConfig.from_pretrained(args.xphonebert_model, **xpb_options)
    xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, **xpb_options)
    if text_hidden_size(qwen_config) != descriptor["text_hidden_size"]:
        raise SystemExit("Loaded Qwen text hidden size does not match the descriptor")
    if getattr(xpb_config, "hidden_size", None) != 768:
        raise SystemExit("Loaded XPhoneBERT config does not have hidden_size=768")
    rows = []
    for split in ("train", "validation", "test"):
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing split: {path}")
        rows.extend(load_jsonl(path))
    units = sorted({unit for row in rows for unit in row["ipa"].split()})
    if not units:
        raise SystemExit("No segmented IPA units found in the experiment splits")
    unk_id = xpb_tokenizer.unk_token_id
    bad_unk = []
    for row in rows:
        ids = token_ids(xpb_tokenizer(row["ipa"], add_special_tokens=True))
        if unk_id in ids:
            bad_unk.append(row["id"])
            if len(bad_unk) >= 10:
                break
    if bad_unk:
        raise SystemExit(f"XPhoneBERT tokenizer produced UNK for records: {bad_unk}")
    special_tokens = [f"<xpb_ipa_{index:03d}>" for index in range(len(units))]
    mapping = dict(zip(units, special_tokens, strict=True))
    qwen_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens + ["<xpb_ipa>"]})
    prompt_max = {"text": 0, "explicit_ipa": 0, "fusion": 0}
    target_max = 0
    total_max = {"text": 0, "explicit_ipa": 0, "fusion": 0}
    for row in rows:
        explicit = "\n分段 IPA：" + " ".join(mapping[unit] for unit in row["ipa"].split())
        target_ids = token_ids(qwen_tokenizer(row["canonical_text"], add_special_tokens=False))
        target_max = max(target_max, len(target_ids))
        for condition, section in (("text", ""), ("explicit_ipa", explicit), ("fusion", "")):
            length = len(chat_ids(qwen_tokenizer, descriptor, row["variant_text"], section))
            prompt_max[condition] = max(prompt_max[condition], length)
            total_max[condition] = max(total_max[condition], length + len(target_ids) + 1)
    qwen_context_limit = context_limit(qwen_config)
    if qwen_context_limit and max(total_max.values()) > qwen_context_limit:
        raise SystemExit(f"Prompt plus target exceeds Qwen context: {max(total_max.values())} > {qwen_context_limit}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ipa_token_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "data_dir": str(args.data_dir),
        "model_descriptor_sha256": digest(args.model_descriptor.read_text(encoding="utf-8")),
        "xphonebert": {"model": args.xphonebert_model, "revision": args.xphonebert_revision, "hidden_size": xpb_config.hidden_size},
        "qwen": {"model": descriptor["model_path_or_repo"], "revision": descriptor["immutable_revision"], "text_hidden_size": text_hidden_size(qwen_config), "context_limit": qwen_context_limit},
        "ipa_units": len(units),
        "ipa_unit_vocabulary_sha256": digest("\0".join(units)),
        "xphonebert_unk_records": 0,
        "prompt_token_max": prompt_max,
        "target_token_max": target_max,
        "max_new_tokens": round_up(target_max),
        "required_max_length": round_up(max(total_max.values())),
        "conditions": ["baseline", "explicit_ipa", "fusion"],
    }
    (args.output_dir / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
