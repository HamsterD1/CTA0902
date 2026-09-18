#!/usr/bin/env python3
"""Full-parameter Text-only SFT used to create the frozen common Qwen base."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data import encode_supervised, load_descriptor, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-descriptor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepspeed", type=Path, required=True, help="ZeRO-3 CPU offload config for single-H100 full SFT")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None, help="Deterministic stratified subset for the required overfit gate")
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


class RestorationDataset:
    def __init__(self, rows, tokenizer, descriptor, max_length: int):
        self.rows = rows
        self.examples = []
        for row in rows:
            item = encode_supervised(tokenizer, descriptor, row)
            if len(item["input_ids"]) > max_length:
                raise ValueError(f"{row['id']} exceeds fixed max length; refusing to truncate")
            self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CausalCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        import torch

        maximum = max(len(row["input_ids"]) for row in features)
        input_ids, labels, masks = [], [], []
        for row in features:
            padding = maximum - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            masks.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def stratified_subset(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        key = row["sample_type"] + ":" + ",".join(row.get("language_tags", []))
        buckets.setdefault(key, []).append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < limit:
                chosen.append(buckets[key].pop())
    return chosen


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        raise SystemExit("Install requirements-h100.pip in cta-xphonebert") from error
    if not torch.cuda.is_available():
        raise SystemExit("Text SFT requires CUDA")
    descriptor = load_descriptor(args.base_descriptor)
    options = {"revision": descriptor["immutable_revision"], "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(descriptor["model_path_or_repo"], **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", **options
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_rows = stratified_subset(load_jsonl(args.data_dir / "train.jsonl"), args.train_limit, args.seed)
    train = RestorationDataset(train_rows, tokenizer, descriptor, args.max_length)
    validation = RestorationDataset(load_jsonl(args.data_dir / "validation.jsonl"), tokenizer, descriptor, args.max_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "condition": "text_baseline_full_sft",
        "base_descriptor": str(args.base_descriptor),
        "base_descriptor_contents": descriptor,
        "train_records": len(train),
        "validation_records": len(validation),
        "full_parameter_sft": True,
        "max_steps": args.max_steps,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    set_seed(args.seed)
    training = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=str(args.deepspeed),
        remove_unused_columns=False,
        max_steps=args.max_steps,
    )
    trainer = Trainer(model=model, args=training, train_dataset=train, eval_dataset=validation, data_collator=CausalCollator(tokenizer))
    trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir / "tokenizer")


if __name__ == "__main__":
    main()
