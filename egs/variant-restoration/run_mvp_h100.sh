#!/usr/bin/env bash
# Run the Text, IPA, and Text+IPA full-fine-tuning ablations on one H100 80GB GPU.
set -euo pipefail

data_dir=${1:-data/variant-restoration/mvp-v1/seq2seq-v1}
# Keep reports and run metadata in the repository.  The checkpoint archive is
# placed on the large-volume mount after each condition has completed.
output_root=${2:-exp/variant-restoration/mvp-v1}
checkpoint_root=${CHECKPOINT_ROOT:-/cpt_dlt/variant-restoration/mvp-v1}
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

    # Trainer needs local checkpoints while it selects the best model.  Once
    # testing is complete, archive each complete checkpoint (model, optimizer,
    # scheduler and RNG state) to the large-volume mount and leave a symlink
    # at the original path, so resume paths continue to work.
    archive_dir="${checkpoint_root}/${condition}"
    mkdir -p "${archive_dir}"
    for checkpoint in "${output_dir}"/checkpoint-*; do
        [[ -L "${checkpoint}" ]] && continue
        [[ -e "${checkpoint}" ]] || continue
        checkpoint_name=$(basename "${checkpoint}")
        mv "${checkpoint}" "${archive_dir}/${checkpoint_name}"
        ln -s "${archive_dir}/${checkpoint_name}" "${checkpoint}"
    done

    # save_model() writes the final selected model into output_dir.  Move only
    # that loadable model bundle; reports, predictions and trainer state stay
    # in the repository output directory.
    final_model_dir="${archive_dir}/final-model"
    mkdir -p "${final_model_dir}"
    for artifact in config.json generation_config.json model.safetensors pytorch_model.bin \
        tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json \
        spiece.model sentencepiece.bpe.model; do
        [[ -L "${output_dir}/${artifact}" ]] && continue
        [[ -e "${output_dir}/${artifact}" ]] || continue
        mv "${output_dir}/${artifact}" "${final_model_dir}/${artifact}"
        ln -s "${final_model_dir}/${artifact}" "${output_dir}/${artifact}"
    done
done
