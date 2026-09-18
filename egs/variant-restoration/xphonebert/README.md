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

## H100 Model Descriptor

On the H100 host, inspect the supplied text-SFT checkpoint before downloading
or training anything. The model path provided for this experiment is
`/data/models/Qwen3.5-9B`; substitute its immutable internal artifact revision.

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

## Preflight

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
