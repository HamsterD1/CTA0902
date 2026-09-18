#!/usr/bin/env python3
"""Create a versioned Qwen model descriptor on the H100 host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path-or-repo", required=True)
    parser.add_argument("--revision", required=True, help="Immutable commit SHA or internal artifact revision")
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--user-template", required=True, help="Must contain {variant_text} and {ipa_section}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.revision.lower() in {"main", "master", "latest"}:
        raise SystemExit("--revision must be immutable, not a moving branch")
    if "{variant_text}" not in args.user_template or "{ipa_section}" not in args.user_template:
        raise SystemExit("--user-template must contain {variant_text} and {ipa_section}")
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install the H100 requirements before inspecting the model") from error
    options = {"revision": args.revision, "trust_remote_code": False, "local_files_only": args.local_files_only}
    config = AutoConfig.from_pretrained(args.model_path_or_repo, **options)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_repo, **options)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size != 5120:
        raise SystemExit(f"Expected Qwen hidden_size=5120, got {hidden_size!r}")
    descriptor = {
        "model_path_or_repo": args.model_path_or_repo,
        "immutable_revision": args.revision,
        "model_type": config.model_type,
        "hidden_size": hidden_size,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "chat_template_sha256": digest(tokenizer.chat_template or ""),
        "prompt_contract": {"system_prompt": args.system_prompt, "user_template": args.user_template},
    }
    descriptor["prompt_contract_sha256"] = digest(json.dumps(descriptor["prompt_contract"], ensure_ascii=False, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(descriptor, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
