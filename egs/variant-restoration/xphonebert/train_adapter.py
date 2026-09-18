#!/usr/bin/env python3
"""Train the frozen-Qwen Explicit IPA or XPhoneBERT Fusion adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter_data import AdapterCollator, AdapterDataset
from adapter_model import ExplicitIpaModel, FusionModel
from data import load_descriptor, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("explicit_ipa", "fusion"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-sft-descriptor", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xphonebert-model", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        raise SystemExit("Install requirements-h100.pip in cta-xphonebert") from error
    if not torch.cuda.is_available():
        raise SystemExit("Adapter training requires CUDA")
    descriptor = load_descriptor(args.text_sft_descriptor)
    preflight = json.loads((args.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ipa_map = json.loads((args.preflight_dir / "ipa_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = list(ipa_map.values()) + ["<xpb_ipa>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    qwen = AutoModelForCausalLM.from_pretrained(
        descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa"
    )
    qwen.config.use_cache = False
    xpb_tokenizer = None
    if args.condition == "explicit_ipa":
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ipa_map.values()] + [tokenizer.convert_tokens_to_ids("<xpb_ipa>")]
        if any(token_id == tokenizer.unk_token_id for token_id in token_ids):
            raise SystemExit("Failed to add all atomic IPA special tokens")
        model = ExplicitIpaModel(qwen, token_ids, tokenizer.pad_token_id)
    else:
        if args.xphonebert_model is None:
            raise SystemExit("--xphonebert-model is required for fusion")
        xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
        xpb = AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
        model = FusionModel(qwen, xpb)
    model.cuda()
    max_length = preflight["required_max_length"]
    train = AdapterDataset(load_jsonl(args.data_dir / "train.jsonl"), tokenizer, descriptor, args.condition, ipa_map, max_length)
    validation = AdapterDataset(load_jsonl(args.data_dir / "validation.jsonl"), tokenizer, descriptor, args.condition, ipa_map, max_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"max_length": max_length, "preflight": preflight, "train_records": len(train), "validation_records": len(validation)}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    set_seed(args.seed)
    training = TrainingArguments(
        output_dir=str(args.output_dir), learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps, bf16=True, tf32=True,
        warmup_ratio=0.03, weight_decay=0.01, max_grad_norm=1.0, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=None, report_to="none",
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training, train_dataset=train, eval_dataset=validation,
        data_collator=AdapterCollator(tokenizer, xpb_tokenizer, args.condition),
    )
    trainer.train()
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")


if __name__ == "__main__":
    main()
