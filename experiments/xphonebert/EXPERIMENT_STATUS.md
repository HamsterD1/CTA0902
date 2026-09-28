# XPhoneBERT Adapter Experiment Status

Updated: 2026-09-28

## Frozen Base

- Frozen Qwen3.5-9B Text-SFT: `checkpoint-5140`, seed 42.
- Deterministic validation (`n=5140`): Top-1 62.2763%, Top-3 74.0272%, Char-Acc 97.6520%.
- Do not rerun this baseline during adapter work; no adapter test decoding has run.

## Explicit IPA

- The deterministic 256-record overfit gate passed at 2,000 steps: mean loss 0.4164725542 to 0.0652614161 (84.33% reduction; gate threshold 80%).

## Fusion

- XPhoneBERT is fixed at `vinai/xphonebert-base` revision `cf2bc63858dec1c03880fa8f764fe2195accb1ab`; the provenance manifest contains only file hashes.
- Its real position-safe input limit is 512 tokens, despite 514 configured positions.
- Lossless encoding uses 510 content tokens plus BOS/EOS, 128-token overlap, and coverage-mean aggregation; IPA text and tokenization are never truncated.
- Preflight found 154 over-limit records: 111 train, 31 validation, and 12 test.
- The longest train IPA (790 content / 792 sequence tokens) passed Fusion forward/backward smoke with all trainable gradients and 25.9 GiB peak allocated memory.

## Stage 1

- Formal Fusion seed-42 trains only the adapters with frozen Qwen/XPhoneBERT, micro-batch 1, and effective global batch 32.
- The launcher supports native PyTorch DDP and performs validation-only checkpoint evaluation.
- It deliberately has no test invocation.

```bash
source /root/autodl-tmp/cta-xphonebert.env
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash egs/variant-restoration/xphonebert/run_fusion_ddp.sh
```

- Select by validation Top-3 then Top-1. Seed 42 must improve Top-3 by at least 1pp over both baseline and Explicit IPA without lowering Top-1 before seeds 43/44.
