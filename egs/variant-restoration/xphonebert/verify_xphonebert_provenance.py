#!/usr/bin/env python3
"""Verify a local XPhoneBERT copy against a trusted fixed-revision snapshot."""

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


def files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_dir.is_dir() or not args.reference_dir.is_dir():
        raise SystemExit("--model-dir and --reference-dir must be directories")
    current = {path.relative_to(args.model_dir).as_posix(): sha256(path) for path in files(args.model_dir)}
    reference = {path.relative_to(args.reference_dir).as_posix(): sha256(path) for path in files(args.reference_dir)}
    missing = sorted(set(reference) - set(current))
    extra = sorted(set(current) - set(reference))
    mismatched = sorted(name for name in set(current) & set(reference) if current[name] != reference[name])
    report = {
        "repo_id": "vinai/xphonebert-base",
        "revision": args.revision,
        "model_dir": str(args.model_dir),
        "reference_dir": str(args.reference_dir),
        "files": current,
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "verified": not missing and not extra and not mismatched,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("revision", "verified", "missing", "extra", "mismatched")}, indent=2))
    if not report["verified"]:
        raise SystemExit("Local XPhoneBERT files do not match the supplied fixed-revision snapshot")


if __name__ == "__main__":
    main()
