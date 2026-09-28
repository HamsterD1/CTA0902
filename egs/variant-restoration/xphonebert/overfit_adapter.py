#!/usr/bin/env python3
"""Deterministic adapter-only overfit gate for the frozen-Qwen conditions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from adapter_data import AdapterCollator, AdapterDataset
from adapter_model import ExplicitIpaModel, FusionModel
from data import load_descriptor, load_jsonl


def stratified_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Pick a deterministic sample while retaining each observed sample type/tag bucket."""
    buckets: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for row in rows:
        key = row["sample_type"], tuple(row.get("language_tags", []))
        buckets.setdefault(key, []).append(row)
    if count < len(buckets):
        raise ValueError(f"--records={count} cannot cover {len(buckets)} strata")
    selected: list[dict] = []
    remainder: list[dict] = []
    for index, key in enumerate(sorted(buckets)):
        bucket = buckets[key]
        random.Random(seed + index).shuffle(bucket)
        selected.append(bucket[0])
        remainder.extend(bucket[1:])
    random.Random(seed).shuffle(remainder)
    return selected + remainder[: count - len(selected)]


def mean_loss(model, collator, examples, batch_size: int, device):
    import torch

    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = {name: value.to(device) for name, value in collator(examples[start : start + batch_size]).items()}
            losses.append(model(**batch).loss.float())
    return torch.stack(losses).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("explicit_ipa", "fusion"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-sft-descriptor", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xphonebert-model", type=Path)
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.records < 1 or args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--records, --steps, and --batch-size must be positive")
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("Adapter overfit gate requires CUDA")
    set_seed(args.seed)
    descriptor = load_descriptor(args.text_sft_descriptor)
    preflight = json.loads((args.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ipa_map = json.loads((args.preflight_dir / "ipa_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(ipa_map.values()) + ["<xpb_ipa>"]})
    qwen = AutoModelForCausalLM.from_pretrained(
        descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa"
    )
    xpb_tokenizer = None
    if args.condition == "explicit_ipa":
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ipa_map.values()] + [tokenizer.convert_tokens_to_ids("<xpb_ipa>")]
        model = ExplicitIpaModel(qwen, token_ids, tokenizer.pad_token_id)
    else:
        if args.xphonebert_model is None:
            raise SystemExit("--xphonebert-model is required for fusion")
        xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
        xpb = AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
        model = FusionModel(qwen, xpb)
    device = torch.device("cuda")
    model.to(device)
    rows = stratified_rows(load_jsonl(args.data_dir / "train.jsonl"), args.records, args.seed)
    dataset = AdapterDataset(rows, tokenizer, descriptor, args.condition, ipa_map, preflight["required_max_length"])
    collator = AdapterCollator(tokenizer, xpb_tokenizer, args.condition)
    initial_loss = mean_loss(model, collator, dataset.examples, args.batch_size, device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.01)
    order = list(range(len(dataset)))
    losses = []
    model.train()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0:
            random.Random(args.seed + step - 1).shuffle(order)
        start = ((step - 1) * args.batch_size) % len(order)
        indices = [order[(start + offset) % len(order)] for offset in range(args.batch_size)]
        batch = {name: value.to(device) for name, value in collator([dataset[index] for index in indices]).items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            losses.append({"step": step, "train_loss": loss.item()})
    final_loss = mean_loss(model, collator, dataset.examples, args.batch_size, device)
    result = {
        "condition": args.condition,
        "seed": args.seed,
        "records": args.records,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initial_mean_loss": initial_loss,
        "final_mean_loss": final_loss,
        "loss_reduction": (initial_loss - final_loss) / initial_loss if initial_loss else None,
        "passes_80_percent_reduction": final_loss <= 0.2 * initial_loss,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "sample_ids": [row["id"] for row in rows],
        "losses": losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passes_80_percent_reduction"]:
        raise SystemExit("Overfit gate failed: final mean loss did not fall by at least 80%")


if __name__ == "__main__":
    main()
