#!/usr/bin/env bash
# Run the Text, IPA, and Text+IPA full-fine-tuning ablations on one H100 80GB GPU.
set -euo pipefail

data_dir=${1:-data/variant-restoration/mvp-v1/seq2seq-v1}
output_root=${2:-/cpt_dlt/variant-restoration/mvp-v1}
model_name=${MODEL_NAME:-google/mt5-base}

for condition in text ipa text_ipa; do
    output_dir="${output_root}/${condition}"
    python egs/variant-restoration/train.py \
        --data-dir "${data_dir}" \
        --condition "${condition}" \
        --output-dir "${output_dir}" \
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
        --predictions "${output_dir}/test_predictions.jsonl" \
        > "${output_dir}/test_report.json"
done
