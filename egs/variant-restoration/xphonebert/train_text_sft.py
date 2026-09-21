#!/usr/bin/env python3
"""Full-parameter Text-only SFT used to create the frozen common Qwen base."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
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
    parser.add_argument("--overfit-save-steps", type=int, default=25, help="Full resumable checkpoint interval for the overfit gate")
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


def warmup_steps(records: int, micro_batch_size: int, gradient_accumulation_steps: int, epochs: int, max_steps: int) -> int:
    if max_steps > 0:
        total_steps = max_steps
    else:
        batches_per_epoch = math.ceil(records / micro_batch_size)
        updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
        total_steps = updates_per_epoch * epochs
    return math.ceil(total_steps * 0.03)


def last_complete_model_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        has_weights = any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").is_file()
        if match and has_weights and (path / "config.json").is_file() and (path / "trainer_state.json").is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))[1]


def checkpoint_progress(checkpoint: Path | None) -> tuple[int, float]:
    if checkpoint is None:
        return 0, 0.0
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state.get("global_step", 0)), float(state.get("epoch", 0.0))


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as error:
        raise SystemExit("Install requirements-h100.pip in cta-xphonebert") from error
    if not torch.cuda.is_available():
        raise SystemExit("Text SFT requires CUDA")
    descriptor = load_descriptor(args.base_descriptor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = last_complete_model_checkpoint(args.output_dir)
    recovered_steps, recovered_epochs = checkpoint_progress(resume_checkpoint)
    options = {"revision": descriptor["immutable_revision"], "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(descriptor["model_path_or_repo"], **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_source = resume_checkpoint or descriptor["model_path_or_repo"]
    model_options = options if resume_checkpoint is None else {"local_files_only": True, "trust_remote_code": False}
    model = AutoModelForCausalLM.from_pretrained(
        model_source, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **model_options
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_rows = stratified_subset(load_jsonl(args.data_dir / "train.jsonl"), args.train_limit, args.seed)
    train = RestorationDataset(train_rows, tokenizer, descriptor, args.max_length)
    is_overfit_gate = args.train_limit is not None
    validation = None
    if not is_overfit_gate:
        validation = RestorationDataset(load_jsonl(args.data_dir / "validation.jsonl"), tokenizer, descriptor, args.max_length)
    effective_max_steps = args.max_steps - recovered_steps if args.max_steps > 0 else -1
    effective_epochs = max(0.0, args.epochs - recovered_epochs) if args.max_steps <= 0 else args.epochs
    if args.max_steps > 0 and effective_max_steps <= 0:
        raise SystemExit("The complete recovery checkpoint has already reached --max-steps")
    if args.max_steps <= 0 and effective_epochs <= 0:
        raise SystemExit("The complete recovery checkpoint has already reached --epochs")
    calculated_warmup_steps = warmup_steps(
        len(train), args.micro_batch_size, args.gradient_accumulation_steps, effective_epochs, effective_max_steps
    )
    run_config = vars(args) | {
        "condition": "text_baseline_full_sft",
        "base_descriptor": str(args.base_descriptor),
        "base_descriptor_contents": descriptor,
        "train_records": len(train),
        "validation_records": len(validation) if validation is not None else 0,
        "full_parameter_sft": True,
        "run_kind": "overfit_gate" if is_overfit_gate else "formal_text_sft",
        "max_steps": args.max_steps,
        "warmup_ratio": 0.03,
        "warmup_steps": calculated_warmup_steps,
        "overfit_save_steps": args.overfit_save_steps if is_overfit_gate else None,
        "checkpoint_mode": "model_only",
        "recovery_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "recovery_resets_optimizer_scheduler": resume_checkpoint is not None,
        "recovered_steps": recovered_steps,
        "recovered_epochs": recovered_epochs,
        "effective_max_steps": effective_max_steps,
        "effective_epochs": effective_epochs,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    set_seed(args.seed)

    class StopOnOverfitSuccess(TrainerCallback):
        def __init__(self, required_reduction: float = 0.80):
            self.required_reduction = required_reduction
            self.first_loss: float | None = None

        def on_train_begin(self, args, state, control, **kwargs):
            losses = [entry["loss"] for entry in state.log_history if entry.get("loss", 0) > 0]
            if losses:
                self.first_loss = losses[0]
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is None or loss <= 0:
                return control
            if self.first_loss is None:
                self.first_loss = loss
            elif 1 - loss / self.first_loss >= self.required_reduction:
                control.should_training_stop = True
            return control

    training = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=effective_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        warmup_steps=calculated_warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="no" if is_overfit_gate else "epoch",
        save_strategy="steps" if is_overfit_gate else "epoch",
        save_steps=args.overfit_save_steps,
        save_total_limit=2 if is_overfit_gate else None,
        save_only_model=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=str(args.deepspeed),
        remove_unused_columns=False,
        max_steps=effective_max_steps,
    )
    callbacks = [StopOnOverfitSuccess()] if is_overfit_gate else None
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=CausalCollator(tokenizer),
        callbacks=callbacks,
    )
    if resume_checkpoint is not None:
        print(f"Restarting from model-only checkpoint: {resume_checkpoint}")
        print("Optimizer and scheduler state are intentionally reset.")
    trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir / "tokenizer")


if __name__ == "__main__":
    main()
