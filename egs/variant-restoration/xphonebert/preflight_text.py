#!/usr/bin/env python3
"""Length and prompt-contract check before full-parameter Text SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import encode_supervised, load_descriptor, load_jsonl
from preflight import digest, round_up
from model_contract import context_limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements-h100.pip in cta-xphonebert") from error
    descriptor = load_descriptor(args.base_descriptor)
    options = {"revision": descriptor["immutable_revision"], "local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(descriptor["model_path_or_repo"], **options)
    tokenizer = AutoTokenizer.from_pretrained(descriptor["model_path_or_repo"], **options)
    maximum = 0
    target_max = 0
    count = 0
    for split in ("train", "validation", "test"):
        for row in load_jsonl(args.data_dir / f"{split}.jsonl"):
            item = encode_supervised(tokenizer, descriptor, row)
            maximum = max(maximum, len(item["input_ids"]))
            target_max = max(target_max, sum(token != -100 for token in item["labels"]))
            count += 1
    limit = context_limit(config)
    if limit and maximum > limit:
        raise SystemExit(f"Text prompt plus target exceeds Qwen context: {maximum} > {limit}")
    result = {
        "records": count,
        "base_descriptor_sha256": digest(args.base_descriptor.read_text(encoding="utf-8")),
        "prompt_contract_sha256": descriptor["prompt_contract_sha256"],
        "context_limit": limit,
        "max_sequence_tokens": maximum,
        "required_max_length": round_up(maximum),
        "target_token_max": target_max,
        "max_new_tokens": round_up(target_max),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
