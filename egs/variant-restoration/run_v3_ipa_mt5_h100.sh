#!/usr/bin/env bash
# Prepare v3 IPA triples and fine-tune mT5-base on one H100 80GB GPU.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

input=${1:-data/拟音清洗/v3/training/拟音还原_含恒等样本_训练样本.json}
artifact_dir=${2:-exp/variant-restoration/v3-mt5-ipa}
checkpoint_root=${CHECKPOINT_ROOT:-/cpt_dlt/variant-restoration/v3-mt5-ipa}
data_dir=${DATA_DIR:-"${checkpoint_root}/prepared-data"}
model_name=${MODEL_NAME:-google/mt5-base}
identity_sample_ratio=${IDENTITY_SAMPLE_RATIO:-0.10}

mkdir -p "${artifact_dir}"
python egs/variant-restoration/prepare_v3_ipa_data.py \
    --input "${input}" --output-dir "${data_dir}" \
    --identity-sample-ratio "${identity_sample_ratio}" --seed 42 \
    | tee "${artifact_dir}/data_manifest.json"

python egs/variant-restoration/train_v3_ipa_mt5.py \
    --data-dir "${data_dir}" --output-dir "${checkpoint_root}/model" \
    --artifact-dir "${artifact_dir}" --model-name "${model_name}" \
    --max-source-length 512 --max-target-length 256 --num-train-epochs 5 \
    --per-device-train-batch-size 8 --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 4 --learning-rate 3e-5 --weight-decay 0.01 \
    --warmup-ratio 0.05 --generation-num-beams 4 --eval-steps 500 --save-steps 500 --seed 42

python egs/variant-restoration/evaluate.py \
    --references "${data_dir}/test.jsonl" --predictions "${artifact_dir}/test_predictions.jsonl" \
    --bucket-field sample_type > "${artifact_dir}/test_report.json"
