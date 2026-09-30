#!/usr/bin/env python3
"""Two-step deterministic gradient reachability check for Fusion warm-start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter_data import AdapterCollator, AdapterDataset
from adapter_model import FusionModel
from data import load_descriptor, load_jsonl
from overfit_adapter import stratified_rows


MODULES = ("resampler", "projector", "fusion")


def group_norm(module, attribute: str) -> tuple[float, bool]:
    import torch

    values = []
    finite = True
    for parameter in module.parameters():
        value = getattr(parameter, attribute)
        if value is None:
            continue
        value = value.detach().float()
        values.append(value.square().sum())
        finite = finite and bool(torch.isfinite(value).all())
    if not values:
        return 0.0, finite
    return torch.stack(values).sum().sqrt().item(), finite


def parameter_snapshots(model: FusionModel) -> dict[str, list]:
    return {
        name: [parameter.detach().float().cpu().clone() for parameter in getattr(model, name).parameters()]
        for name in MODULES
    }


def parameter_deltas(model: FusionModel, before: dict[str, list]) -> dict[str, float]:
    import torch

    result = {}
    for name in MODULES:
        values = [
            (parameter.detach().float().cpu() - initial).square().sum()
            for parameter, initial in zip(getattr(model, name).parameters(), before[name], strict=True)
        ]
        result[name] = torch.stack(values).sum().sqrt().item() if values else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--text-sft-descriptor", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--xphonebert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.records < 1:
        raise SystemExit("--records must be positive")

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("Fusion reachability calibration requires CUDA")
    set_seed(args.seed)
    descriptor = load_descriptor(args.text_sft_descriptor)
    preflight = json.loads((args.preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ipa_map = json.loads((args.preflight_dir / "ipa_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        descriptor.get("tokenizer_path_or_repo", descriptor["model_path_or_repo"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    xpb_tokenizer = AutoTokenizer.from_pretrained(args.xphonebert_model, local_files_only=True, trust_remote_code=False)
    qwen = AutoModelForCausalLM.from_pretrained(
        descriptor["model_path_or_repo"], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa"
    )
    xpb = AutoModel.from_pretrained(args.xphonebert_model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    model = FusionModel(qwen, xpb).cuda()
    rows = stratified_rows(load_jsonl(args.data_dir / "train.jsonl"), args.records, args.seed)
    dataset = AdapterDataset(rows, tokenizer, descriptor, "fusion", ipa_map, preflight["required_max_length"])
    collator = AdapterCollator(tokenizer, xpb_tokenizer, "fusion")
    batch = {name: value.cuda() for name, value in collator([dataset[0]]).items()}

    model.eval()
    with torch.inference_mode():
        base_embeddings = model.qwen.get_input_embeddings()(batch["input_ids"])
        fusion_embeddings = model.embeddings(
            batch["input_ids"], batch["ipa_input_ids"], batch["ipa_attention_mask"], batch["ipa_window_token_index"],
            batch["ipa_window_example_index"], batch["ipa_sequence_lengths"], batch["variant_mask"]
        )
        base_logits = model.qwen(inputs_embeds=base_embeddings, attention_mask=batch["attention_mask"], use_cache=False).logits
        fusion_logits = model.qwen(inputs_embeds=fusion_embeddings, attention_mask=batch["attention_mask"], use_cache=False).logits
    step_zero = {
        "embedding_max_abs_difference": (base_embeddings - fusion_embeddings).abs().max().item(),
        "logit_max_abs_difference": (base_logits - fusion_logits).abs().max().item(),
        "embeddings_exact": bool(torch.equal(base_embeddings, fusion_embeddings)),
        "logits_exact": bool(torch.equal(base_logits, fusion_logits)),
    }

    before = parameter_snapshots(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate, weight_decay=0.01)
    model.train()
    steps = []
    losses = []
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss_finite = bool(torch.isfinite(loss))
        if not loss_finite:
            raise SystemExit(f"Non-finite loss at step {step}")
        loss.backward()
        gradients = {}
        for name in MODULES:
            norm, finite = group_norm(getattr(model, name), "grad")
            gradients[name] = {"norm": norm, "finite": finite}
        if not all(item["finite"] for item in gradients.values()):
            raise SystemExit(f"Non-finite gradient at step {step}")
        optimizer.step()
        parameter_finite = all(
            bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters() if parameter.requires_grad
        )
        if not parameter_finite:
            raise SystemExit(f"Non-finite parameter at step {step}")
        losses.append(loss.item())
        steps.append({"step": step, "loss": loss.item(), "gradients": gradients, "parameters_finite": parameter_finite})

    deltas = parameter_deltas(model, before)
    loss_limit = max(1.0, losses[0] * 10.0)
    checks = {
        "step_zero_exact_text_base": step_zero["embeddings_exact"] and step_zero["logits_exact"],
        "all_modules_nonzero_gradient_after_two_steps": all(steps[-1]["gradients"][name]["norm"] > 0.0 for name in MODULES),
        "all_modules_parameter_changed_after_two_steps": all(deltas[name] > 0.0 for name in MODULES),
        "loss_nondivergent": all(torch.isfinite(torch.tensor(losses))) and max(losses) <= loss_limit,
        "all_trainable_parameters_finite": all(step["parameters_finite"] for step in steps),
    }
    report = {
        "condition": "fusion_warmstart_reachability",
        "seed": args.seed,
        "records": args.records,
        "learning_rate": args.learning_rate,
        "calibration_sample_id": rows[0]["id"],
        "step_zero": step_zero,
        "steps": steps,
        "parameter_delta_norms": deltas,
        "loss_nondivergence_limit": loss_limit,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Fusion reachability calibration failed")


if __name__ == "__main__":
    main()
