#!/usr/bin/env python3
"""Deterministic beam-3 evaluation for frozen-Qwen IPA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter_data import AdapterDataset
from adapter_model import ExplicitIpaModel, FusionModel
from data import decode_continuations, load_descriptor, load_jsonl, metric_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("explicit_ipa", "fusion"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-sft-descriptor", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xphonebert-model", type=Path)
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements-h100.pip in cta-xphonebert") from error
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
        ids = [tokenizer.convert_tokens_to_ids(token) for token in ipa_map.values()] + [tokenizer.convert_tokens_to_ids("<xpb_ipa>")]
        model = ExplicitIpaModel(qwen, ids, tokenizer.pad_token_id)
    else:
        if args.xphonebert_model is None:
            raise SystemExit("--xphonebert-model is required for fusion")
        xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
        model = FusionModel(qwen, AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False))
    model.load_adapter(args.checkpoint)
    model.cuda().eval()
    dataset = AdapterDataset(load_jsonl(args.data_dir / f"{args.split}.jsonl"), tokenizer, descriptor, args.condition, ipa_map, preflight["required_max_length"])
    records = []
    with torch.inference_mode():
        for item, row in zip(dataset.examples, dataset.rows, strict=True):
            input_ids = torch.tensor([item["input_ids"][:item["prompt_length"]]], device="cuda")
            mask = torch.ones_like(input_ids)
            variant = torch.tensor([item["variant_mask"][:item["prompt_length"]]], device="cuda")
            if args.condition == "explicit_ipa":
                embeddings = model.adapter.apply(input_ids, model.qwen.get_input_embeddings(), model.fallback_token_id)
            else:
                ipa = xpb_tokenizer([row["ipa"]], return_tensors="pt")
                embeddings = model.embeddings(input_ids, ipa["input_ids"].cuda(), ipa["attention_mask"].cuda(), variant)
            generated = model.qwen.generate(inputs_embeds=embeddings, attention_mask=mask, num_beams=3, num_return_sequences=3,
                do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            predictions = decode_continuations(tokenizer, generated, input_ids.shape[1])
            records.append({"id": row["id"], "variant_text": row["variant_text"], "ipa": row["ipa"], "canonical_text": row["canonical_text"],
                "sample_type": row["sample_type"], "language_tags": row.get("language_tags", []), "predictions": predictions,
                "text_token_count": input_ids.shape[1], "ipa_phoneme_count": len(row["ipa"].split()), "resampler_length": input_ids.shape[1]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    report = metric_report(records)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
