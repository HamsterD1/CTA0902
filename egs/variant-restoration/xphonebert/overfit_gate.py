#!/usr/bin/env python3
"""Fail the experiment when the mandated tiny-set overfit gate is not met."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-loss-reduction", type=float, default=0.80)
    args = parser.parse_args()
    history = json.loads(args.trainer_state.read_text(encoding="utf-8")).get("log_history", [])
    losses = [entry["loss"] for entry in history if "loss" in entry and entry["loss"] > 0]
    if len(losses) < 2:
        raise SystemExit("Overfit run did not emit at least two training loss points")
    reduction = 1 - losses[-1] / losses[0]
    report = {"first_loss": losses[0], "last_loss": losses[-1], "loss_reduction": reduction, "required_loss_reduction": args.required_loss_reduction, "passed": reduction >= args.required_loss_reduction}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Overfit gate failed; do not start formal training")


if __name__ == "__main__":
    main()
