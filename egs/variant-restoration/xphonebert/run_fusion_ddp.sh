#!/usr/bin/env bash
set -euo pipefail

# Frozen-Qwen/XPhoneBERT Fusion only. Validation is allowed; test is deliberately absent.
data_dir=${DATA_DIR:-experiments/xphonebert/data-split-v1}
descriptor=${TEXT_SFT_DESCRIPTOR:-/root/autodl-tmp/autodl-local/text-sft-descriptor.json}
preflight_dir=${PREFLIGHT_DIR:-experiments/xphonebert/preflight}
xphonebert_model=${XPHONEBERT_MODEL:-/root/autodl-tmp/models/xphonebert-base}
checkpoint_root=${CHECKPOINT_ROOT:-/root/autodl-tmp/variant-restoration/xphonebert}
seed=${SEED:-42}
nproc_per_node=${NPROC_PER_NODE:-2}
effective_batch_size=${EFFECTIVE_BATCH_SIZE:-32}
micro_batch_size=1

[[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC_PER_NODE must be positive" >&2; exit 1; }
[[ "$effective_batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "EFFECTIVE_BATCH_SIZE must be positive" >&2; exit 1; }
(( effective_batch_size % (micro_batch_size * nproc_per_node) == 0 )) || {
  echo "effective batch must be divisible by micro batch times world size" >&2
  exit 1
}
gradient_accumulation_steps=$((effective_batch_size / (micro_batch_size * nproc_per_node)))
run_dir="$checkpoint_root/fusion/seed-$seed"

test -f "$descriptor"
test -f "$preflight_dir/preflight.json"
test -f "$preflight_dir/ipa_token_map.json"
test -f "$xphonebert_model/REVISION"
test "$(tr -d '\n' < "$xphonebert_model/REVISION")" = "cf2bc63858dec1c03880fa8f764fe2195accb1ab"

python - "$preflight_dir/preflight.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
xpb = report["xphonebert"]
assert xpb["revision"] == "cf2bc63858dec1c03880fa8f764fe2195accb1ab"
assert xpb["position_safe_max_input_tokens"] == 512
assert xpb["chunking"] == {"overlap_tokens": 128, "aggregation": "coverage_mean", "preserves_bos_eos": True}
PY

mkdir -p "$run_dir"
python -m torch.distributed.run --standalone --nproc_per_node="$nproc_per_node" \
  egs/variant-restoration/xphonebert/train_adapter.py \
  --condition fusion \
  --data-dir "$data_dir" \
  --text-sft-descriptor "$descriptor" \
  --preflight-dir "$preflight_dir" \
  --xphonebert-model "$xphonebert_model" \
  --output-dir "$run_dir" \
  --micro-batch-size "$micro_batch_size" \
  --gradient-accumulation-steps "$gradient_accumulation_steps" \
  --seed "$seed"

max_new_tokens=$(python -c "import json; print(json.load(open('$preflight_dir/preflight.json'))['max_new_tokens'])")
for checkpoint in "$run_dir"/checkpoint-*; do
  test -d "$checkpoint" || continue
  python egs/variant-restoration/xphonebert/evaluate_adapter.py \
    --condition fusion \
    --data-dir "$data_dir" \
    --text-sft-descriptor "$descriptor" \
    --preflight-dir "$preflight_dir" \
    --xphonebert-model "$xphonebert_model" \
    --checkpoint "$checkpoint" \
    --split validation \
    --max-new-tokens "$max_new_tokens" \
    --output-dir "$run_dir/validation/$(basename "$checkpoint")"
done
