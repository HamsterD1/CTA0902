#!/usr/bin/env python3
"""Run one forward/backward Fusion smoke on the longest real IPA sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter_data import AdapterCollator, AdapterDataset
from adapter_model import FusionModel
from data import load_descriptor, load_jsonl
from xphonebert_chunks import chunk_ipa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-sft-descriptor", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--xphonebert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("Chunked Fusion smoke requires CUDA")
    descriptor = load_descriptor(args.text_sft_descriptor)
    preflight = json.loads((args.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ipa_map = json.loads((args.preflight_dir / "ipa_token_map.json").read_text(encoding="utf-8"))
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False
    )
    if qwen_tokenizer.pad_token_id is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
    rows = load_jsonl(args.data_dir / "train.jsonl")
    selected = max(rows, key=lambda row: chunk_ipa(xpb_tokenizer, row["ipa"]).content_token_count)
    plan = chunk_ipa(xpb_tokenizer, selected["ipa"])
    dataset = AdapterDataset([selected], qwen_tokenizer, descriptor, "fusion", ipa_map, preflight["required_max_length"])
    batch = {name: value.cuda() for name, value in AdapterCollator(qwen_tokenizer, xpb_tokenizer, "fusion")(dataset.examples).items()}
    qwen = AutoModelForCausalLM.from_pretrained(
        descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa"
    )
    xpb = AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    model = FusionModel(qwen, xpb).cuda().train()
    result = model(**batch)
    if not torch.isfinite(result.loss):
        raise SystemExit("Chunked Fusion smoke loss is not finite")
    result.loss.backward()
    trainable = {name: parameter.grad is not None for name, parameter in model.named_parameters() if parameter.requires_grad}
    report = {
        "id": selected["id"],
        "ipa_content_token_count": plan.content_token_count,
        "ipa_sequence_token_count": plan.sequence_token_count,
        "max_xphonebert_input_tokens": 512,
        "window_count": len(plan.windows),
        "window_input_lengths": [len(window.input_ids) for window in plan.windows],
        "loss": result.loss.item(),
        "all_trainable_grad_present": all(trainable.values()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_trainable_grad_present"]:
        raise SystemExit("Chunked Fusion smoke did not reach every trainable parameter")


if __name__ == "__main__":
    main()
