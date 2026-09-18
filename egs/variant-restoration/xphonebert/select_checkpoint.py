#!/usr/bin/env python3
"""Choose only from validation metrics: Top-3 Recall, then Top-1 exact match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for metrics_path in args.run_dir.glob("validation/checkpoint-*/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["all"]
        candidates.append({"checkpoint": str(args.run_dir / metrics_path.parent.name), "metrics": metrics})
    if not candidates:
        raise SystemExit("No validation checkpoint metrics found")
    best = max(candidates, key=lambda row: (row["metrics"]["top3_recall"], row["metrics"]["top1_exact_match"]))
    args.output.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
