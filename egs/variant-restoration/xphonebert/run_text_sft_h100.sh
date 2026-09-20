#!/usr/bin/env bash
set -euo pipefail

# Run from the dedicated CTA0902-xphonebert worktree with cta-xphonebert active.
# This recipe is deliberately single-H100.  Multiple visible GPUs make
# Transformers select DataParallel before DeepSpeed initializes the model.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

data_dir=${DATA_DIR:-experiments/xphonebert/data-split-v1}
artifact_root=${ARTIFACT_ROOT:-experiments/xphonebert}
checkpoint_root=${CHECKPOINT_ROOT:-/newdata/dlt/checkpoints/variant-restoration/xphonebert}
base_model=${BASE_MODEL:-/data/models/Qwen3.5-9B}
base_revision=${BASE_REVISION:?Set BASE_REVISION to fingerprint_model.py output}
training_export=${TRAINING_EXPORT:-data/拟音清洗/v3/training/拟音还原_含恒等样本_多语种统一IPA_训练样本.json}
coverage_file=${COVERAGE_FILE:-data/拟音清洗/v3/training/xphonebert_多语种统一IPA_coverage.json}
seed=${SEED:-42}
micro_batch=${MICRO_BATCH_SIZE:-1}
grad_accum=${GRADIENT_ACCUMULATION_STEPS:-32}

mkdir -p "$artifact_root" "$checkpoint_root"
test -f "$training_export" || { echo "Missing TRAINING_EXPORT: $training_export" >&2; exit 1; }
test -f "$coverage_file" || { echo "Missing COVERAGE_FILE: $coverage_file" >&2; exit 1; }
python egs/variant-restoration/xphonebert/prepare_data.py \
  --input "$training_export" --coverage "$coverage_file" \
  --output-dir "$data_dir" --seed "$seed"
python egs/variant-restoration/xphonebert/inspect_model.py \
  --model-path-or-repo "$base_model" --revision "$base_revision" \
  --system-prompt '你是中文拟音文本还原助手。仅输出规范还原文本。' \
  --user-template '拟音文本：{variant_text}{ipa_section}' \
  --output "$artifact_root/base-model-descriptor.json" --local-files-only
python egs/variant-restoration/xphonebert/preflight_text.py \
  --data-dir "$data_dir" --base-descriptor "$artifact_root/base-model-descriptor.json" \
  --output "$artifact_root/text-preflight.json"
max_length=$(python -c "import json; print(json.load(open('$artifact_root/text-preflight.json'))['required_max_length'])")
run_dir="$checkpoint_root/text-baseline/seed-$seed"
overfit_dir="$checkpoint_root/text-baseline/overfit-seed-$seed"
python egs/variant-restoration/xphonebert/train_text_sft.py \
  --data-dir "$data_dir" --base-descriptor "$artifact_root/base-model-descriptor.json" \
  --output-dir "$overfit_dir" --deepspeed egs/variant-restoration/xphonebert/deepspeed-zero2-cpu-offload.json \
  --max-length "$max_length" --micro-batch-size "$micro_batch" --gradient-accumulation-steps "$grad_accum" \
  --train-limit 256 --max-steps 2000 --seed "$seed"
python egs/variant-restoration/xphonebert/overfit_gate.py \
  --trainer-state "$overfit_dir/trainer_state.json" --output "$overfit_dir/overfit_gate.json"
python egs/variant-restoration/xphonebert/train_text_sft.py \
  --data-dir "$data_dir" --base-descriptor "$artifact_root/base-model-descriptor.json" \
  --output-dir "$run_dir" --deepspeed egs/variant-restoration/xphonebert/deepspeed-zero2-cpu-offload.json \
  --max-length "$max_length" --micro-batch-size "$micro_batch" --gradient-accumulation-steps "$grad_accum" --seed "$seed"
for checkpoint in "$run_dir"/checkpoint-*; do
  test -d "$checkpoint" || continue
  python egs/variant-restoration/xphonebert/evaluate_text.py \
    --data-dir "$data_dir" --descriptor "$artifact_root/base-model-descriptor.json" --checkpoint "$checkpoint" \
    --split validation --max-new-tokens "$(python -c "import json; print(json.load(open('$artifact_root/text-preflight.json'))['max_new_tokens'])")" \
    --output-dir "$run_dir/validation/$(basename "$checkpoint")"
done
python egs/variant-restoration/xphonebert/select_checkpoint.py --run-dir "$run_dir" --output "$run_dir/best_validation_checkpoint.json"
best_checkpoint=$(python -c "import json; print(json.load(open('$run_dir/best_validation_checkpoint.json'))['checkpoint'])")
python egs/variant-restoration/xphonebert/create_checkpoint_descriptor.py \
  --checkpoint "$best_checkpoint" --base-descriptor "$artifact_root/base-model-descriptor.json" \
  --training-run-config "$run_dir/run_config.json" --output "$artifact_root/text-sft-descriptor.json"
python egs/variant-restoration/xphonebert/evaluate_text.py \
  --data-dir "$data_dir" --descriptor "$artifact_root/text-sft-descriptor.json" --checkpoint "$best_checkpoint" --split test \
  --max-new-tokens "$(python -c "import json; print(json.load(open('$artifact_root/text-preflight.json'))['max_new_tokens'])")" \
  --output-dir "$run_dir/test"
