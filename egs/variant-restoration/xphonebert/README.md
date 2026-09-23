# Qwen3.5 and XPhoneBERT Restoration Experiments

This is the current runbook for the Qwen3.5 Text Baseline, Explicit IPA, and
XPhoneBERT Fusion comparison. It is independent from the legacy mT5 recipes:
do not overwrite their environments, worktrees, data, or results.

## Current Status

As of 2026-09-23, Stage 0 Text Baseline full-parameter SFT has completed three
epochs on the H100. Training loss decreased stably to roughly 0.01--0.03 with
no reported NaN or gradient instability, while reported validation loss rose
across epochs. This is not a restoration-quality conclusion: beam-3 validation
of the base and epoch-3 checkpoints remains pending. No checkpoint has yet
been selected, and Explicit IPA and Fusion have not started.

The official Stage 0 output root is
`/cpt_dlt/variant-restoration/xphonebert`. `experiments/xphonebert/` stores
splits, descriptors, preflight reports, and final reports only; it must not
store model checkpoints.

## Immutable Contract

| Item | Current value |
| --- | --- |
| Qwen base | `/data/models/Qwen3.5-9B` |
| Qwen revision | `sha256:df3a5598099b4cd3c4e42bcb97ebcf9d2430682957fb699dbab9022bdf846db4` |
| Qwen text config | `Qwen3_5TextConfig`, hidden size 4096, 32 layers, 16 heads |
| Qwen tokenizer | `Qwen2Tokenizer`, vocab size 248,077 |
| System prompt | `你是中文拟音文本还原助手。仅输出规范还原文本。` |
| User template | `拟音文本：{variant_text}{ipa_section}` |
| Prompt hash | `2b565ae2c025fd5ce15945aaa103f3c2b1f8900fc3420a363381923a9890d36d` |
| XPhoneBERT | `vinai/xphonebert-base` at `cf2bc63858dec1c03880fa8f764fe2195accb1ab` |
| Split policy | exact-triple deduplication, then canonical-text-grouped 80/10/10, seed 42 |
| Decoding | deterministic beam search: 3 beams, 3 returns, no sampling |

Do not use an old 5120-dimensional Qwen assumption. The H100 model inspection
resolved the actual text-side width to 4096. Do not change the prompt, its
hash, the retained v3 IPA boundaries, or source training JSON during this
experiment series.

## Data and Evaluation Contract

Stage 0 reads:

```text
data/拟音清洗/v3/training/拟音还原_含恒等样本_多语种统一IPA_训练样本.json
data/拟音清洗/v3/training/xphonebert_多语种统一IPA_coverage.json
```

All 51,752 records, including identity examples, participate in splitting.
`sample_type` is retained. The prepared split contains 41,107 train, 5,140
validation, and 5,142 test records, with 15,341 / 1,922 / 1,932 canonical
targets respectively.

Before training, `preflight_text.py` scans all split prompts and targets. It
rounds required sequence and generation budgets upward and fails on any
overflow; it never truncates or drops a sample. The current Text Baseline
preflight resolved `max_length=768`.

Metrics are top-1 exact match and top-3 exact-match recall at character level.
The normalized variant applies NFC and line-ending normalization only. Select a
checkpoint by validation top-3, then validation top-1 on ties. Run test once,
only for that selected checkpoint.

`evaluate_text.py` supports batched validation with left padding. It preserves
the same deterministic beam-3 contract as single-sample evaluation. Start with
`--batch-size 4` on an otherwise idle 80 GB H100, reduce it after an OOM, and
use the same batch size for every checkpoint comparison. The launcher exposes
this as `EVAL_BATCH_SIZE` (default 1). It emits progress every 10 batches and
writes `progress.json` plus `predictions.partial.jsonl` after every batch;
these partial artifacts remain available for diagnosis after interruption but
are not valid evaluation results. A completed run renames the predictions file
and writes `metrics.json`.

## Stage 0: Text Baseline

Text Baseline supplies `variant_text` only. It is full-parameter Qwen SFT with
bf16, TF32, gradient checkpointing, single-H100 execution, and DeepSpeed
ZeRO-2 CPU optimizer offload.

| Hyperparameter | Value |
| --- | --- |
| Seed | 42 |
| Epoch limit | 5 |
| Learning rate | 1e-5 |
| Warmup | 3% (193 updates for a fresh 5-epoch run) |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Micro batch / accumulation | 1 / 32 (effective batch 32) |
| Validation and model save | once per epoch |
| Overfit gate | stratified 256 records, at most 2,000 updates, >=80% loss reduction |

DeepSpeed is required despite an 80 GB H100: full-parameter AdamW optimizer
state does not fit safely alongside 9B bf16 weights, gradients, activations,
and workspace. CPU offload trades throughput for memory feasibility.

### Checkpoint and Recovery Policy

Checkpoints use `save_only_model=True`: they include model weights,
`config.json`, and `trainer_state.json`, but deliberately omit DeepSpeed
optimizer/scheduler state. This avoids writing roughly 100 GB optimizer
checkpoints to the shared filesystem. A complete checkpoint must contain all
three artifacts, including at least one `.safetensors` file.

On restart, training loads the newest complete model-only checkpoint from the
same output directory, resets optimizer and scheduler state, and limits the
remaining epoch/update budget so the experiment does not exceed five epochs or
the overfit gate's 2,000-update limit. This is model recovery, not bit-exact
optimizer-state resumption. Run long jobs in `tmux` and tee output to
`$CHECKPOINT_ROOT/text-baseline/seed-42/launch.log`.

### H100 Launch

Run from `/newdata/dlt/CTA0902-xphonebert` after activating
`cta-xphonebert`. Keep these exports in the same shell:

```bash
export CUDA_VISIBLE_DEVICES=2
export BASE_MODEL=/data/models/Qwen3.5-9B
export CHECKPOINT_ROOT=/cpt_dlt/variant-restoration/xphonebert
export ARTIFACT_ROOT=experiments/xphonebert

python egs/variant-restoration/xphonebert/fingerprint_model.py \
  --model-dir "$BASE_MODEL" \
  --output "$ARTIFACT_ROOT/base-model-fingerprint.json"

export BASE_REVISION="$(python -c "import json; print(json.load(open('experiments/xphonebert/base-model-fingerprint.json'))['immutable_revision'])")"

OVERFIT_SAVE_STEPS=100 \
bash egs/variant-restoration/xphonebert/run_text_sft_h100.sh
```

The launcher fixes visibility to a single GPU, uses `torchrun` to avoid MPI
discovery, runs the overfit gate, trains Text SFT, validates saved epoch
checkpoints with beam-3, selects the best checkpoint, writes
`experiments/xphonebert/text-sft-descriptor.json`, then evaluates test once.
The messages about `torch_dtype` deprecation, CUDA 12.8 versus a PyTorch CUDA
12.1 build, and missing optional attention kernels are known warnings. The
latter two reduce performance but do not invalidate results.

## Stage 1: Frozen IPA Conditions

Run Stage 1 only after Stage 0 selects a Text SFT checkpoint. Both conditions
reuse that exact selected checkpoint, the canonical split, prompt contract,
and beam-3 evaluator.

| Condition | Text input | Trainable components |
| --- | --- | --- |
| Explicit IPA | `variant_text` plus atomic segmented IPA tokens | only 120 new IPA token embeddings and `[IPA]` separator embedding |
| Fusion | `variant_text`; IPA goes only to XPhoneBERT | Resampler, Projector, `alpha_raw` |

Fusion freezes both Text-SFT Qwen and XPhoneBERT. Its contract is one pre-LN
cross-attention layer with text queries `4096 -> 768`, XPhoneBERT Key/Value
width 768, 12 heads, projector `768 -> 2048 -> GELU -> 4096`, and a residual
coefficient `tanh(alpha_raw)` initialized at zero. Only prompt-token positions
are enhanced. Explicit IPA must have fewer trainable parameters than Fusion.

XPhoneBERT has not yet been fully transferred to the H100 host because that
host cannot currently fetch it from Hugging Face. This blocks Stage 1 but not
Text Baseline.

Stage 1 uses adapter learning rate 1e-4, the same 3% warmup, weight decay
0.01, clipping 1.0, and at most five epochs. Fusion seed 42 may advance to
seeds 43/44 only if validation top-3 improves by at least 1.0 percentage point
over both Baseline and Explicit IPA without lowering top-1. Final claims need
positive mean test top-3 improvement with a paired-bootstrap 95% CI excluding
zero.

## Verification Checklist

- Verify input embedding width 4096 when loading Qwen with
  `AutoModelForCausalLM` on the H100.
- Retain `base-model-fingerprint.json`, model descriptor, split manifest,
  text preflight report, and selected Text SFT descriptor in
  `experiments/xphonebert/`.
- Do not report training loss as restoration quality. Use validation beam-3
  metrics to decide whether to continue beyond an epoch checkpoint.
- Do not start Explicit IPA or Fusion before the final Text Baseline checkpoint
  and its descriptor exist.
