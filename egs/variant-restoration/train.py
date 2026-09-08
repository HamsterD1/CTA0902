#!/usr/bin/env python3
"""Full fine-tuning entry point for the phonetic variant restoration MVP."""
import argparse
import json
from pathlib import Path


SPECIAL_TOKENS = ["[VARIANT]", "[IPA]"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("text", "ipa", "text_ipa"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/mt5-base")
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=25)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--generation-num-beams", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def character_accuracy(reference, prediction):
    if not reference:
        return float(not prediction)
    matches = sum(left == right for left, right in zip(reference, prediction))
    return matches / max(len(reference), len(prediction))


def main():
    args = parse_args()
    try:
        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise SystemExit("Install egs/variant-restoration/requirements-h100.txt before training.") from error

    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this full fine-tuning recipe. Use --allow-cpu only for API smoke tests.")
    files = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing Seq2Seq triple files: {', '.join(missing)}")

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    raw_dataset = load_dataset("json", data_files={name: str(path) for name, path in files.items()})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize(batch):
        inputs = batch["inputs"]
        source_texts = inputs[args.condition] if isinstance(inputs, dict) else [item[args.condition] for item in inputs]
        encoded = tokenizer(source_texts, max_length=args.max_source_length, truncation=True)
        encoded["labels"] = tokenizer(
            text_target=batch["canonical_text"], max_length=args.max_target_length, truncation=True
        )["input_ids"]
        return encoded

    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset["train"].column_names)

    def decode_generated_ids(token_ids):
        """Decode generated sequences after Trainer's cross-batch padding.

        Trainer pads predictions from differently sized generation batches with
        -100 before metrics are computed. SentencePiece cannot decode that
        sentinel value, unlike label decoding where it is conventionally
        replaced with the tokenizer pad ID.
        """
        token_ids = np.asarray(token_ids)
        token_ids = np.where(token_ids == -100, tokenizer.pad_token_id, token_ids)
        if token_ids.size and (token_ids.min() < 0 or token_ids.max() >= len(tokenizer)):
            raise ValueError("Generated token ID is outside the tokenizer vocabulary.")
        return [value.strip() for value in tokenizer.batch_decode(token_ids, skip_special_tokens=True)]

    def compute_metrics(prediction_output):
        predictions, labels = prediction_output
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        decoded_predictions = decode_generated_ids(predictions)
        decoded_labels = [value.strip() for value in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        exact = sum(prediction == label for prediction, label in zip(decoded_predictions, decoded_labels))
        char_acc = sum(character_accuracy(label, prediction) for prediction, label in zip(decoded_predictions, decoded_labels))
        count = max(1, len(decoded_labels))
        return {"exact_match": exact / count, "character_accuracy": char_acc / count}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.generation_num_beams,
        # Matching step strategies permit selecting the best checkpoint while
        # avoiding a multi-gigabyte save after every tiny-data epoch.
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        # Exact match is initially often tied at zero; validation loss gives
        # a meaningful checkpoint ordering until generations become usable.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    test_result = trainer.predict(tokenized["test"], metric_key_prefix="test")
    trainer.save_metrics("test", test_result.metrics)
    trainer.save_state()

    decoded = decode_generated_ids(test_result.predictions)
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item, prediction in zip(raw_dataset["test"], decoded):
            handle.write(json.dumps({
                "id": item["id"],
                "condition": args.condition,
                "input": item["inputs"][args.condition],
                "ipa": item["ipa"],
                "reference": item["canonical_text"],
                "prediction": prediction.strip(),
                "variant_type": item["variant_type"],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
