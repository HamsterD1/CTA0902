#!/usr/bin/env bash
set -euo pipefail

# Requires the selected Text SFT checkpoint and a locally transferred XPhoneBERT.
# Use torchrun DDP for multi-GPU training. Evaluation stays single-process.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

data_dir=${DATA_DIR:-experiments/xphonebert/data-split-v1}
artifact_root=${ARTIFACT_ROOT:-experiments/xphonebert}
checkpoint_root=${CHECKPOINT_ROOT:-/cpt_dlt/variant-restoration/xphonebert}
xphonebert_model=${XPHONEBERT_MODEL:-/data/models/xphonebert-base}
xphonebert_revision=${XPHONEBERT_REVISION:-cf2bc63858dec1c03880fa8f764fe2195accb1ab}
effective_batch=${EFFECTIVE_BATCH_SIZE:-32}
seed=${SEED:-42}
text_sft_checkpoint=${TEXT_SFT_CHECKPOINT:-$checkpoint_root/text-baseline/seed-$seed/checkpoint-5140}
nproc_per_node=${NPROC_PER_NODE:-1}

[[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC_PER_NODE must be a positive integer" >&2; exit 1; }
train_launcher=(python)
if (( nproc_per_node > 1 )); then
  train_launcher=(torchrun --standalone --nproc_per_node="$nproc_per_node")
fi

test -d "$text_sft_checkpoint"
test -f "$artifact_root/base-model-descriptor.json"
test -f "$checkpoint_root/text-baseline/seed-$seed/run_config.json"
python egs/variant-restoration/xphonebert/create_checkpoint_descriptor.py \
  --checkpoint "$text_sft_checkpoint" --base-descriptor "$artifact_root/base-model-descriptor.json" \
  --training-run-config "$checkpoint_root/text-baseline/seed-$seed/run_config.json" \
  --output "$artifact_root/text-sft-descriptor.json"
test -f "$artifact_root/text-preflight.json"
python egs/variant-restoration/xphonebert/preflight.py \
  --data-dir "$data_dir" --model-descriptor "$artifact_root/text-sft-descriptor.json" \
  --xphonebert-model "$xphonebert_model" --xphonebert-revision "$xphonebert_revision" \
  --output-dir "$artifact_root/preflight" --local-files-only
max_new=$(python -c "import json; print(json.load(open('$artifact_root/preflight/preflight.json'))['max_new_tokens'])")
python egs/variant-restoration/xphonebert/calibrate_fusion_batch.py \
  --data-dir "$data_dir" --text-sft-descriptor "$artifact_root/text-sft-descriptor.json" \
  --preflight-dir "$artifact_root/preflight" --xphonebert-model "$xphonebert_model" \
  --effective-batch-size "$effective_batch" --world-size "$nproc_per_node" \
  --output "$artifact_root/fusion-batch-calibration.json"
micro_batch=$(python -c "import json; print(json.load(open('$artifact_root/fusion-batch-calibration.json'))['micro_batch_size'])")
grad_accum=$(python -c "import json; print(json.load(open('$artifact_root/fusion-batch-calibration.json'))['gradient_accumulation_steps'])")
for condition in explicit_ipa fusion; do
  run_dir="$checkpoint_root/$condition/seed-$seed"
  options=(--condition "$condition" --data-dir "$data_dir" --text-sft-descriptor "$artifact_root/text-sft-descriptor.json" --preflight-dir "$artifact_root/preflight" --output-dir "$run_dir" --micro-batch-size "$micro_batch" --gradient-accumulation-steps "$grad_accum" --seed "$seed")
  smoke_options=(--condition "$condition" --data-dir "$data_dir" --text-sft-descriptor "$artifact_root/text-sft-descriptor.json" --preflight-dir "$artifact_root/preflight")
  eval_options=(--condition "$condition" --data-dir "$data_dir" --text-sft-descriptor "$artifact_root/text-sft-descriptor.json" --preflight-dir "$artifact_root/preflight" --max-new-tokens "$max_new")
  if [[ "$condition" == fusion ]]; then
    options+=(--xphonebert-model "$xphonebert_model")
    eval_options+=(--xphonebert-model "$xphonebert_model")
    smoke_options+=(--xphonebert-model "$xphonebert_model")
  fi
  python egs/variant-restoration/xphonebert/smoke_adapter.py "${smoke_options[@]}" --output "$run_dir/diagnostics/smoke.json"
  "${train_launcher[@]}" egs/variant-restoration/xphonebert/train_adapter.py "${options[@]}"
  for checkpoint in "$run_dir"/checkpoint-*; do
    test -d "$checkpoint" || continue
    python egs/variant-restoration/xphonebert/evaluate_adapter.py "${eval_options[@]}" --checkpoint "$checkpoint" --split validation --output-dir "$run_dir/validation/$(basename "$checkpoint")"
  done
  python egs/variant-restoration/xphonebert/select_checkpoint.py --run-dir "$run_dir" --output "$run_dir/best_validation_checkpoint.json"
done
