#!/usr/bin/env python3
"""Forward, shape, frozen-parameter, and gradient smoke check for IPA adapters."""

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
    parser.add_argument("--xphonebert-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("Adapter smoke test requires CUDA")
    descriptor = load_descriptor(args.text_sft_descriptor)
    preflight = json.loads((args.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ipa_map = json.loads((args.preflight_dir / "ipa_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(ipa_map.values()) + ["<xpb_ipa>"]})
    qwen = AutoModelForCausalLM.from_pretrained(descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    xpb_tokenizer = None
    if args.condition == "explicit_ipa":
        token_ids = [tokenizer.convert_tokens_to_ids(value) for value in ipa_map.values()] + [tokenizer.convert_tokens_to_ids("<xpb_ipa>")]
        model = ExplicitIpaModel(qwen, token_ids, tokenizer.pad_token_id)
    else:
        if args.xphonebert_model is None:
            raise SystemExit("--xphonebert-model is required for fusion")
        xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
        xpb = AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
        model = FusionModel(qwen, xpb)
    rows = load_jsonl(args.data_dir / "train.jsonl")
    selected, seen = [], set()
    for row in rows:
        key = row["sample_type"] + ":" + ",".join(row.get("language_tags", []))
        if key not in seen:
            selected.append(row)
            seen.add(key)
    dataset = AdapterDataset(selected, tokenizer, descriptor, args.condition, ipa_map, preflight["required_max_length"])
    batch = {name: value.cuda() for name, value in AdapterCollator(tokenizer, xpb_tokenizer, args.condition)(dataset.examples).items()}
    model.cuda().train()
    result = model(**batch)
    if not torch.isfinite(result.loss):
        raise SystemExit("Smoke loss is not finite")
    result.loss.backward()
    frozen = {name: parameter.grad is None for name, parameter in model.named_parameters() if not parameter.requires_grad}
    trainable = {name: parameter.grad is not None for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not all(frozen.values()) or not all(trainable.values()):
        raise SystemExit("Gradient contract failed; inspect gradient_check.json")
    report = {
        "condition": args.condition, "batch_records": len(selected), "loss": result.loss.item(),
        "logits_shape": list(result.logits.shape), "all_frozen_grad_none": all(frozen.values()),
        "all_trainable_grad_present": all(trainable.values()), "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
