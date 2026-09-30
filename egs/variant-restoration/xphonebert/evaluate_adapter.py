#!/usr/bin/env python3
"""Deterministic beam-3 evaluation for frozen-Qwen IPA adapters."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from adapter_data import AdapterCollator, AdapterDataset
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
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
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
    collator = AdapterCollator(tokenizer, xpb_tokenizer, args.condition)
    example_count = len(dataset.examples) if args.max_examples is None else min(args.max_examples, len(dataset.examples))
    records = [None] * example_count
    by_prompt_length = defaultdict(list)
    for index, item in enumerate(dataset.examples[:example_count]):
        by_prompt_length[item["prompt_length"]].append(index)

    def evaluate_batch(indices: list[int]) -> None:
        items = [dataset.examples[index] for index in indices]
        rows = [dataset.rows[index] for index in indices]
        prompt_length = items[0]["prompt_length"]
        if any(item["prompt_length"] != prompt_length for item in items):
            raise ValueError("Evaluation batches must have a common prompt length")
        batch = {name: value.cuda() for name, value in collator(items).items()}
        input_ids = batch["input_ids"][:, :prompt_length]
        mask = batch["attention_mask"][:, :prompt_length]
        variant = batch["variant_mask"][:, :prompt_length]
        if args.condition == "explicit_ipa":
            embeddings = model.adapter.apply(input_ids, model.qwen.get_input_embeddings(), model.fallback_token_id)
        else:
            embeddings = model.embeddings(
                input_ids, batch["ipa_input_ids"], batch["ipa_attention_mask"], batch["ipa_window_token_index"],
                batch["ipa_window_example_index"], batch["ipa_sequence_lengths"], variant
            )
        generated = model.qwen.generate(inputs_embeds=embeddings, attention_mask=mask, num_beams=3, num_return_sequences=3,
            do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        predictions = decode_continuations(tokenizer, generated, input_ids.shape[1])
        expected_predictions = 3 * len(indices)
        if len(predictions) != expected_predictions:
            raise RuntimeError(f"Expected {expected_predictions} beam predictions, got {len(predictions)}")
        for local_index, (record_index, row) in enumerate(zip(indices, rows, strict=True)):
            records[record_index] = {"id": row["id"], "variant_text": row["variant_text"], "ipa": row["ipa"], "canonical_text": row["canonical_text"],
                "sample_type": row["sample_type"], "language_tags": row.get("language_tags", []), "predictions": predictions[3 * local_index:3 * (local_index + 1)],
                "text_token_count": input_ids.shape[1], "ipa_phoneme_count": len(row["ipa"].split()), "resampler_length": input_ids.shape[1]}

    with torch.inference_mode():
        for indices in by_prompt_length.values():
            for start in range(0, len(indices), args.batch_size):
                evaluate_batch(indices[start:start + args.batch_size])
    if any(record is None for record in records):
        raise RuntimeError("Evaluation did not produce a prediction for every record")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    report = metric_report(records)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
