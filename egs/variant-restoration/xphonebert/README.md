# XPhoneBERT Fusion Experiment

This is an independent Qwen/XPhoneBERT recipe. It neither rewrites the retained
v3 IPA files nor shares output directories with the mT5 experiments.

## Local Data Contract

Create the immutable split once. Do not regenerate it after a run begins.

```bash
python egs/variant-restoration/xphonebert/prepare_data.py \
  --input data/拟音清洗/v3/training/拟音还原_含恒等样本_多语种统一IPA_训练样本.json \
  --coverage data/拟音清洗/v3/training/xphonebert_多语种统一IPA_coverage.json \
  --output-dir experiments/xphonebert/data-split-v1
```

The generated manifest records the input hash, exact deduplication, and a
canonical-text-disjoint 80/10/10 split. It keeps every identity record.

## Stage 0: Text-Only SFT

The supplied `/data/models/Qwen3.5-9B` checkpoint is an untrained base model.
Text-only full-parameter SFT creates the shared frozen checkpoint for all
subsequent comparisons.

The legacy mT5 environment pins `transformers==4.45.0`, which cannot load a
checkpoint declaring `model_type: qwen3_5`. Create a separate XPhoneBERT
environment rather than upgrading that existing environment in place:

```bash
conda create -n cta-xphonebert python=3.12 -y
conda activate cta-xphonebert
python -m pip install -r egs/variant-restoration/xphonebert/requirements-h100.pip
```

Fingerprint the complete local base artifact and run Text SFT:

```bash
python egs/variant-restoration/xphonebert/fingerprint_model.py \
  --model-dir /data/models/Qwen3.5-9B \
  --output experiments/xphonebert/base-model-fingerprint.json
export BASE_REVISION="$(python -c "import json; print(json.load(open('experiments/xphonebert/base-model-fingerprint.json'))['immutable_revision'])")"
bash egs/variant-restoration/xphonebert/run_text_sft_h100.sh
```

The launcher creates the split, validates full prompt-plus-target lengths,
runs the required 256-record / 2,000-step overfit gate (requiring at least an
80% training-loss reduction), then runs 5-epoch full SFT with ZeRO-2 CPU
optimizer offload. It evaluates each epoch with deterministic beam-3 on
validation, and only then evaluates the selected checkpoint once on test. The selected checkpoint descriptor is written to
`experiments/xphonebert/text-sft-descriptor.json`.

```bash
python egs/variant-restoration/xphonebert/inspect_model.py \
  --model-path-or-repo /data/models/Qwen3.5-9B \
  --revision '<immutable-revision>' \
  --system-prompt '<original-text-sft-system-prompt>' \
  --user-template '请将以下拟音文本还原为规范文本，仅输出还原结果。\n拟音文本：{variant_text}{ipa_section}' \
  --output experiments/xphonebert/model-descriptor.json \
  --local-files-only
```

`--revision` must be an immutable commit or internal artifact revision, not
`main`, `master`, or `latest`. Use the original text-SFT prompt verbatim when
one exists. The command verifies the 5,120-dimensional Qwen contract and
writes tokenizer/chat-template hashes.

## Stage 1: Frozen IPA Adapters

The audited XPhoneBERT tokenizer revision is
`cf2bc63858dec1c03880fa8f764fe2195accb1ab`. Run preflight before every new
descriptor or data split:

```bash
python egs/variant-restoration/xphonebert/preflight.py \
  --data-dir experiments/xphonebert/data-split-v1 \
  --model-descriptor experiments/xphonebert/model-descriptor.json \
  --xphonebert-revision cf2bc63858dec1c03880fa8f764fe2195accb1ab \
  --output-dir experiments/xphonebert/preflight \
  --local-files-only
```

It fails on an IPA `[UNK]`, hidden-size mismatch, non-120-unit vocabulary,
or any prompt-plus-target that exceeds the Qwen context. It also emits the
atomic IPA token map required by the Explicit IPA condition and the fixed
`max_new_tokens` / context budget.

## Fusion Contract

`modeling.py` implements the agreed Stage 1 modules:

- Frozen XPhoneBERT (768d) is cross-attended by a 5,120d-to-768d Query
  projection with pre-LayerNorm, 12 heads, and no dropout.
- The projector is `768 -> 2048 -> GELU -> 5120`.
- `tanh(alpha_raw)` starts at zero and only changes variant prompt positions.
- Explicit IPA uses standalone embedding rows for the 120 new atomic tokens;
  Qwen's original embedding matrix and all Qwen parameters remain frozen.

Before starting a GPU run, archive `model-descriptor.json`, `preflight.json`,
`ipa_token_map.json`, and the prepared split manifest inside the run directory.

Run both IPA adapter conditions after the Text SFT launcher completes:

```bash
export XPHONEBERT_MODEL=/data/models/xphonebert-base
bash egs/variant-restoration/xphonebert/run_adapters_h100.sh
```

The launcher first calibrates the largest Fusion micro-batch on a P100-length
sample, derives gradient accumulation for an effective batch of 32, then uses
those identical values for Explicit IPA and Fusion. It trains only the agreed
adapter parameters, evaluates every epoch with deterministic beam-3, and
selects checkpoints from validation only.
