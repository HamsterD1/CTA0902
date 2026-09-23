#!/usr/bin/env python3
"""Deterministic beam-3 evaluation for a text-only Qwen restoration checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import load_descriptor, load_jsonl, metric_report, prompt_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.progress_every_batches < 1:
        raise SystemExit("--progress-every-batches must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements-h100.pip in cta-xphonebert") from error
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires CUDA")
    descriptor = load_descriptor(args.descriptor)
    tokenizer = AutoTokenizer.from_pretrained(descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
    ).cuda().eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    records = []
    rows = load_jsonl(args.data_dir / f"{args.split}.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_predictions = args.output_dir / "predictions.partial.jsonl"
    progress_path = args.output_dir / "progress.json"
    with partial_predictions.open("w", encoding="utf-8") as output:
        with torch.inference_mode():
            for offset in range(0, len(rows), args.batch_size):
                batch_rows = rows[offset:offset + args.batch_size]
                prompts = [prompt_ids(tokenizer, descriptor, row) for row in batch_rows]
                batch = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
                ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                outputs = model.generate(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    num_beams=3,
                    num_return_sequences=3,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                decoded = tokenizer.batch_decode(outputs[:, ids.shape[1]:], skip_special_tokens=True)
                for index, row in enumerate(batch_rows):
                    predictions = decoded[index * 3:(index + 1) * 3]
                    record = {
                        "id": row["id"], "variant_text": row["variant_text"], "ipa": row["ipa"],
                        "canonical_text": row["canonical_text"], "sample_type": row["sample_type"],
                        "language_tags": row.get("language_tags", []), "predictions": [value.strip() for value in predictions],
                        "text_token_count": len(prompts[index]), "ipa_phoneme_count": len(row["ipa"].split()),
                        "resampler_length": len(prompts[index]),
                    }
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                completed = offset + len(batch_rows)
                progress_path.write_text(
                    json.dumps({"completed": completed, "total": len(rows), "batch_size": args.batch_size}) + "\n",
                    encoding="utf-8",
                )
                batch_number = offset // args.batch_size + 1
                if batch_number == 1 or batch_number % args.progress_every_batches == 0 or completed == len(rows):
                    print(f"Evaluated {completed}/{len(rows)} validation records", flush=True)
    partial_predictions.replace(args.output_dir / "predictions.jsonl")
    metrics = metric_report(records)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
