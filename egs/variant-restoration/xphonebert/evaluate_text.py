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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
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
    records = []
    with torch.inference_mode():
        for row in load_jsonl(args.data_dir / f"{args.split}.jsonl"):
            ids = torch.tensor([prompt_ids(tokenizer, descriptor, row)], device=model.device)
            outputs = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                num_beams=3,
                num_return_sequences=3,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            predictions = tokenizer.batch_decode(outputs[:, ids.shape[1]:], skip_special_tokens=True)
            records.append({
                "id": row["id"], "variant_text": row["variant_text"], "ipa": row["ipa"],
                "canonical_text": row["canonical_text"], "sample_type": row["sample_type"],
                "language_tags": row.get("language_tags", []), "predictions": [value.strip() for value in predictions],
                "text_token_count": ids.shape[1], "ipa_phoneme_count": len(row["ipa"].split()), "resampler_length": ids.shape[1],
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    metrics = metric_report(records)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
