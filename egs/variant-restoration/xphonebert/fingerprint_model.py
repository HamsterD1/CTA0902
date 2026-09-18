#!/usr/bin/env python3
"""Produce an immutable content fingerprint for a local model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suffixes = (".json", ".safetensors", ".bin", ".model")
    files = sorted(path for path in args.model_dir.iterdir() if path.is_file() and path.name.endswith(suffixes))
    if not files:
        raise SystemExit("No model artifact files found")
    entries = [{"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in files]
    manifest = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    result = {"model_dir": str(args.model_dir), "files": entries, "immutable_revision": "sha256:" + hashlib.sha256(manifest.encode()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["immutable_revision"])


if __name__ == "__main__":
    main()
