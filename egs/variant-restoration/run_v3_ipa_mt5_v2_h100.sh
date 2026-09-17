#!/usr/bin/env bash
# Audit real mT5 token lengths, then run a larger-batch v3 IPA-only experiment.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

input=${1:-data/拟音清洗/v3/training/拟音还原_含恒等样本_训练样本.json}
artifact_dir=${2:-exp/variant-restoration/v3-mt5-ipa-v2-b12}
checkpoint_root=${CHECKPOINT_ROOT:-/cpt_dlt/variant-restoration/v3-mt5-ipa-v2-b12}
data_dir=${DATA_DIR:-"${checkpoint_root}/prepared-data"}
model_name=${MODEL_NAME:-/model_dlt/mt5-base}
identity_sample_ratio=${IDENTITY_SAMPLE_RATIO:-0.10}
max_source_length=${MAX_SOURCE_LENGTH:-768}
max_target_length=${MAX_TARGET_LENGTH:-384}
train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-12}
eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE:-24}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}
epochs=${NUM_TRAIN_EPOCHS:-10}

mkdir -p "${artifact_dir}"
python egs/variant-restoration/audit_v3_ipa_token_lengths.py \
    --input "${input}" --model-name "${model_name}" \
    --output "${artifact_dir}/token_length_audit.json" \
    --selected-source-length "${max_source_length}" --selected-target-length "${max_target_length}"

python egs/variant-restoration/prepare_v3_ipa_data.py \
    --input "${input}" --output-dir "${data_dir}" \
    --identity-sample-ratio "${identity_sample_ratio}" --seed 42 \
    | tee "${artifact_dir}/data_manifest.json"

python egs/variant-restoration/train_v3_ipa_mt5.py \
    --data-dir "${data_dir}" --output-dir "${checkpoint_root}/model" \
    --artifact-dir "${artifact_dir}" --model-name "${model_name}" \
    --max-source-length "${max_source_length}" --max-target-length "${max_target_length}" \
    --num-train-epochs "${epochs}" --per-device-train-batch-size "${train_batch_size}" \
    --per-device-eval-batch-size "${eval_batch_size}" \
    --gradient-accumulation-steps "${gradient_accumulation_steps}" \
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.05 \
    --generation-num-beams 4 --eval-steps 500 --save-steps 500 --seed 42

python egs/variant-restoration/evaluate.py \
    --references "${data_dir}/test.jsonl" --predictions "${artifact_dir}/test_predictions.jsonl" \
    --bucket-field sample_type > "${artifact_dir}/test_report.json"
