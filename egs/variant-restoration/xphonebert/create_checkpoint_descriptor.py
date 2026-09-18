#!/usr/bin/env python3
"""Bind a selected text-SFT checkpoint to its base model and prompt contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data import load_descriptor


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-descriptor", type=Path, required=True)
    parser.add_argument("--training-run-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load_descriptor(args.base_descriptor)
    required = ("config.json", "trainer_state.json")
    missing = [name for name in required if not (args.checkpoint / name).is_file()]
    if missing:
        raise SystemExit(f"Checkpoint lacks required files: {missing}")
    hashes = {path.name: file_hash(path) for path in sorted(args.checkpoint.glob("*.json"))}
    descriptor = base | {
        "model_path_or_repo": str(args.checkpoint),
        "tokenizer_path_or_repo": base.get("tokenizer_path_or_repo", base["model_path_or_repo"]),
        "immutable_revision": "text-sft:" + file_hash(args.training_run_config),
        "checkpoint_kind": "text_sft_full_parameter",
        "base_model_descriptor_sha256": file_hash(args.base_descriptor),
        "training_run_config_sha256": file_hash(args.training_run_config),
        "checkpoint_json_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(descriptor, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
