#!/usr/bin/env bash
# Run the Text, IPA, and Text+IPA full-fine-tuning ablations on one H100 80GB GPU.
set -euo pipefail

# This recipe is deliberately single-H100.  When several GPUs are visible,
# Transformers falls back to DataParallel, which is both unnecessary here and
# unstable with this mT5 setup.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

data_dir=${1:-data/variant-restoration/mvp-v1/seq2seq-v1}
# Keep reports and run metadata in the repository.  Trainer writes checkpoints
# and the final model directly to the large-volume mount from the first step.
artifact_root=${2:-exp/variant-restoration/mvp-v1}
checkpoint_root=${CHECKPOINT_ROOT:-/cpt_dlt/variant-restoration/mvp-v1}
model_name=${MODEL_NAME:-google/mt5-base}

for condition in text ipa text_ipa; do
    artifact_dir="${artifact_root}/${condition}"
    checkpoint_dir="${checkpoint_root}/${condition}"
    python egs/variant-restoration/train.py \
        --data-dir "${data_dir}" \
        --condition "${condition}" \
        --output-dir "${checkpoint_dir}" \
        --artifact-dir "${artifact_dir}" \
        --model-name "${model_name}" \
        --num-train-epochs 25 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 32 \
        --gradient-accumulation-steps 2 \
        --learning-rate 3e-5 \
        --weight-decay 0.01 \
        --warmup-ratio 0.1 \
        --generation-num-beams 4 \
        --eval-steps 50 \
        --save-steps 50 \
        --seed 42
    python egs/variant-restoration/evaluate.py \
        --references "${data_dir}/test.jsonl" \
        --predictions "${artifact_dir}/test_predictions.jsonl" \
        > "${artifact_dir}/test_report.json"
done
