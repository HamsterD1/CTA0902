# XPhoneBERT Weak-Residual Fusion, cap=0.1, Seed 42

Status: **training in progress, validation only**, 2026-09-30.

This is the authoritative record for the minimal follow-up to the completed
`fusion-warmstart/seed-42` experiment. It does not replace the historical
warm-start report. No test result exists for this condition.

## Decision Context

The completed Fusion warm-start was learnable but did not beat the frozen
Text-SFT checkpoint. Its best checkpoint was epoch 2 / `checkpoint-2570`:

| Condition | Top-1 | Top-3 | Character accuracy |
| --- | ---: | ---: | ---: |
| Archived Text-SFT checkpoint-5140 | 62.2763% | 74.0272% | 97.6520% |
| Original Fusion epoch 2 | 61.6148% | 73.9300% | 97.5457% |

The compatible paired control showed a small phonetic Top-3 signal but
identity and candidate-ranking damage. Of 97 cases where Text-SFT rank 1 was
correct and Fusion rank 1 was wrong, 74 still retained the gold candidate at
Fusion rank 2 or 3. The next action is therefore one bounded residual change,
not a new representation, data source, prompt, or hyperparameter sweep.

Read `docs/xphonebert-fusion-warmstart-seed42.md` for the five-epoch historical
run and `docs/xphonebert-fusion-paired-diagnosis-seed42.md` for the paired
evidence.

## Frozen Contract

| Item | Frozen value |
| --- | --- |
| Text base | Qwen3.5-9B Text-SFT seed 42 `checkpoint-5140` |
| Text base identity | `text-sft:ae9d4903ac277bda22de1aa62579308f0c0026aa1af87b894beae9e312f1cbfd` |
| XPhoneBERT | `vinai/xphonebert-base` revision `cf2bc63858dec1c03880fa8f764fe2195accb1ab` |
| Split | v3 exact-triple deduplication, canonical-text-grouped 80/10/10, seed 42 |
| Records read | 41,107 train and 5,140 validation; no test read |
| Text / IPA rules | v3 IPA, source JSON, prompt/hash, normalization, and metric rules |
| IPA encoding | 512 total XPhoneBERT tokens, at most 510 content tokens, 128-token overlap, coverage-mean aggregation; no truncation |
| Validation decoding | deterministic beam-3, three returned hypotheses, no sampling, batch size 10 |

Preflight reports 154 overlong IPA records (maximum 792 IPA tokens), zero
unknown XPhoneBERT tokens, text prompt maximum 378 tokens, target maximum 321
tokens, and required model sequence length 1,664. Overlong IPA is encoded
losslessly; no record is truncated or dropped.

Evaluation batch size is protocol-bearing: the linear-attention Qwen model can
change deterministic beam ordering with a different batch layout. Batch 10
remains mandatory for comparisons to the archived baseline.

## Sole Model Change

Qwen and XPhoneBERT remain frozen. The approximately 15.48M trainable adapter
path remains:

```text
Qwen variant-token embeddings
  -> pre-LN 12-head cross-attention resampler (4096 -> 768)
  -> projector (768 -> 2048 -> GELU -> 4096)
  -> residual only at variant-token positions
```

The projector final linear layer remains zero-initialized. Only the residual
coefficient is bounded:

```text
alpha = 0.1 * tanh(alpha_raw)
alpha_raw(0) = atanh(0.5)
alpha(0) = 0.05
embeddings_out = embeddings_text + alpha * projected_phonetics * variant_mask
```

Historical Fusion used `alpha_cap=1.0` and initial alpha 0.5. This condition
uses `alpha_cap=0.1`, preserving a nonzero gradient path while reducing the
maximum injection tenfold. This is the sole architectural change.

## Reachability Calibration

The deterministic 256-record, two-update calibration passed. At step zero,
weak-residual embeddings and logits exactly matched the frozen Text-SFT model.
After two updates, resampler, projector, and gate all had finite nonzero
gradients and parameter changes; loss was finite and nondivergent. This proves
only that the path is connected and stable, not validation quality.

Artifact outside Git:

```text
/root/autodl-tmp/variant-restoration/xphonebert/
  fusion-weak-residual-cap01/seed-42/reachability.json
```

## Training Record

The intended configuration is AdamW, learning rate `1e-4`, 3% warmup, weight
decay `0.01`, gradient clipping `1.0`, BF16 compute with TF32, seed 42, and at
most five epochs. Epoch 1 completed as `micro-batch=1`, accumulation 32
(effective batch 32), saving `checkpoint-1285`.

At the experiment owner's direction, training resumed from this complete
checkpoint as `micro-batch=3`, accumulation 11 (effective batch 33). Trainer
state, optimizer, scheduler, and RNG were restored. This is a **mixed-batch
continuation**, not a pristine five-epoch run under one batch geometry:

```text
epoch 1: b1 / accumulation 32 (effective 32)
continued after checkpoint-1285: b3 / accumulation 11 (effective 33)
```

Do not equate later checkpoints with a fixed-geometry seed-42 replication.
`run_config.json` records the active restart arguments; the epoch-1 checkpoint
retains the earlier Trainer state.

Source is on the persistent system disk at `/root/CTA0902-xphonebert`; models,
split, checkpoints, predictions, logs, and Python environment remain on the
`/root/autodl-tmp` data disk. Weights and run artifacts are never committed.

## Completed Validation: Epoch 1

`checkpoint-1285` was evaluated on all 5,140 validation records using the
frozen deterministic beam-3 / batch-10 protocol:

| Group | Count | Top-1 | Top-3 | Character accuracy |
| --- | ---: | ---: | ---: | ---: |
| All | 5,140 | 61.9066% | 74.0078% | 97.5702% |
| Phonetic | 4,501 | 58.1426% | 71.0953% | 97.3629% |
| Identity | 639 | 88.4194% | 94.5227% | 99.1553% |

All-record Top-3 is `-0.0194pp` below archived Text-SFT Top-3 (`74.0272%`)
and closer than original Fusion epoch 1 (`73.4047%`). Top-1 remains below the
base. This does not select a model or authorize test evaluation.

Artifact outside Git:

```text
/root/autodl-tmp/variant-restoration/xphonebert/
  fusion-weak-residual-cap01/seed-42/validation-b10/
    checkpoint-1285/metrics.json
```

## Next Safe Action

1. Let the active b3/acc11 continuation reach saved checkpoints without
   concurrent GPU generation.
2. Evaluate each checkpoint only on validation with beam 3 and batch 10.
3. Record Top-1, Top-3, and character accuracy against Text-SFT and original
   Fusion, retaining the mixed-batch caveat.
4. Do not run test, seed 43/44, extra epochs, or alter the cap, IPA, split,
   prompt/hash, decoding, or metric rules from interim validation.
5. A later decision may choose a separately predeclared fixed-geometry
   replication; this continuation alone is not eligible for a test claim.

## Operational Notes

- Active log: `/root/autodl-tmp/variant-restoration/xphonebert/fusion-weak-residual-cap01/seed-42/train-b3-resume.log`.
- `--fusion-alpha-cap` and `--resume-from-checkpoint` are source additions for
  this condition.
- Missing frozen Qwen/XPhoneBERT keys when loading a compact adapter checkpoint
  are expected. The resume batch-size warning is expected provenance for the
  mixed-batch continuation.
