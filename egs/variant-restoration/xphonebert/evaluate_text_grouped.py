#!/usr/bin/env python3
"""Text-SFT evaluator with Fusion-compatible same-prompt-length batch cohorts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data import decode_continuations, load_descriptor, load_jsonl, metric_report, prompt_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    descriptor = load_descriptor(args.descriptor)
    tokenizer = AutoTokenizer.from_pretrained(descriptor["tokenizer_path_or_repo"], local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False).cuda().eval()
    rows = load_jsonl(args.data_dir / "validation.jsonl")
    prompts = [prompt_ids(tokenizer, descriptor, row) for row in rows]
    cohorts = defaultdict(list)
    for index, prompt in enumerate(prompts):
        cohorts[len(prompt)].append(index)
    records = [None] * len(rows)
    with torch.inference_mode():
        for indices in cohorts.values():
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                ids = torch.tensor([prompts[index] for index in batch_indices], dtype=torch.long, device=model.device)
                outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), num_beams=3, num_return_sequences=3,
                    do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                predictions = decode_continuations(tokenizer, outputs, ids.shape[1])
                for local_index, row_index in enumerate(batch_indices):
                    row = rows[row_index]
                    records[row_index] = {"id": row["id"], "variant_text": row["variant_text"], "ipa": row["ipa"], "canonical_text": row["canonical_text"],
                        "sample_type": row["sample_type"], "language_tags": row.get("language_tags", []), "predictions": predictions[3 * local_index:3 * (local_index + 1)],
                        "text_token_count": len(prompts[row_index]), "ipa_phoneme_count": len(row["ipa"].split()), "resampler_length": len(prompts[row_index])}
    if any(record is None for record in records):
        raise RuntimeError("Evaluation did not produce a prediction for every record")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    report = metric_report(records)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
