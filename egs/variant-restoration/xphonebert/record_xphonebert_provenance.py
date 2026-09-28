#!/usr/bin/env python3
"""Record immutable local XPhoneBERT revision metadata and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision_path = args.model_dir / "REVISION"
    if not revision_path.is_file():
        raise SystemExit("Local XPhoneBERT copy has no REVISION file")
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != args.expected_revision:
        raise SystemExit(f"Expected {args.expected_revision}, found {revision}")
    files = {
        path.relative_to(args.model_dir).as_posix(): sha256(path)
        for path in sorted(args.model_dir.iterdir())
        if path.is_file()
    }
    report = {
        "repo_id": "vinai/xphonebert-base",
        "revision": revision,
        "revision_metadata_file": "REVISION",
        "verification_method": "local_revision_file_and_sha256_manifest",
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": revision, "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
