#!/usr/bin/env python3
"""Find the largest viable Fusion micro-batch on a P100-length example."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from adapter_data import AdapterCollator, AdapterDataset
from adapter_model import FusionModel
from data import load_descriptor, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-sft-descriptor", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--xphonebert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-micro-batch", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("Fusion batch calibration requires CUDA")
    descriptor = load_descriptor(args.text_sft_descriptor)
    preflight = json.loads((args.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ipa_map = json.loads((args.preflight_dir / "ipa_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
    rows = load_jsonl(args.data_dir / "train.jsonl")
    dataset = AdapterDataset(rows, tokenizer, descriptor, "fusion", ipa_map, preflight["required_max_length"])
    ranked = sorted(dataset.examples, key=lambda item: len(item["input_ids"]))
    p100_index = max(0, math.ceil(0.99 * len(ranked)) - 1)
    example = ranked[p100_index]
    qwen = AutoModelForCausalLM.from_pretrained(descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa")
    xpb = AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    model = FusionModel(qwen, xpb).cuda().train()
    collator = AdapterCollator(tokenizer, xpb_tokenizer, "fusion")
    best = 0
    attempts = []
    size = 1
    while size <= args.max_micro_batch:
        torch.cuda.empty_cache()
        try:
            batch = {name: value.cuda() for name, value in collator([example] * size).items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            attempts.append({"micro_batch_size": size, "status": "ok", "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            best = size
            size *= 2
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            attempts.append({"micro_batch_size": size, "status": "oom"})
            break
    if not best:
        raise SystemExit("Fusion P100 sample does not fit micro-batch 1")
    result = {
        "p100_sequence_tokens": len(example["input_ids"]),
        "micro_batch_size": best,
        "effective_batch_size": args.effective_batch_size,
        "gradient_accumulation_steps": math.ceil(args.effective_batch_size / best),
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
