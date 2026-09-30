# XPhoneBERT Fusion Warm-Start Seed 42

Status: completed validation-only experiment, 2026-09-30.

This is the authoritative summary of the Qwen3.5 plus XPhoneBERT Fusion
warm-start run. It records validation evidence only. No Fusion checkpoint has
been selected for test evaluation.

## Decision

The Fusion adapter was structurally learnable but did not improve the frozen
Text-SFT base under the fixed validation protocol. Do not run test evaluation
or seeds 43/44 from this run. Epoch 2 was closest on Top-3, but it still
reduced Top-1 and did not clear the required improvement gate.

## Frozen Inputs

- Text base: Text-SFT seed 42 checkpoint-5140, Qwen hidden size 4096.
- XPhoneBERT: cf2bc63858dec1c03880fa8f764fe2195accb1ab, hidden size 768.
- Split: v3 exact-triple deduplication followed by canonical-text-grouped
  80/10/10 splitting with seed 42.
- Records: 41,107 train and 5,140 validation.
- Decoding: validation only, deterministic beam search with 3 beams and
  3 returned hypotheses. No sampling and no test split.
- Invariants: v3 IPA boundaries, source JSON, prompt/hash, normalization, and
  metric definitions were not changed.

XPhoneBERT encodes IPA losslessly: every window is capped at 512 total input
tokens, therefore carries at most 510 IPA content tokens after BOS and EOS.
Adjacent windows overlap by 128 tokens, and coverage-mean aggregation
reconstructs each original IPA token. The run preflight reported 154 overlong
IPA records, a maximum of 792 IPA tokens, and zero unknown XPhoneBERT tokens.

## Trainable Fusion Path

Qwen and XPhoneBERT remain frozen. The only trainable parameters are the
approximately 15.48M-parameter Fusion adapter:

1. A pre-LN cross-attention resampler projects Qwen text embeddings
   4096 -> 768 and attends to frozen XPhoneBERT states with 12 heads.
2. A projector maps 768 -> 2048 -> GELU -> 4096.
3. A gated residual adds the projected phonetic signal only at the variant-text
   token positions.

The projector final linear layer is zero-initialized. The residual gate is
initialized to tanh(atanh(0.5)) = 0.5 and retained in FP32. Therefore step zero
is exactly equal to the frozen Text-SFT forward pass while gradients can reach
the resampler, projector, and gate.

The two-step deterministic reachability calibration passed:

- step-zero embeddings and logits exactly matched the frozen text base;
- after two optimizer updates all three trainable module groups had finite,
  nonzero gradients and nonzero parameter changes;
- loss stayed finite and nondivergent.

The archived calibration artifact is under the AutoDL local artifact root at
fusion-reachability-warmstart/seed-42.json.

## Training Configuration

| Setting | Value |
| --- | --- |
| Condition | Fusion |
| Seed | 42 |
| Epochs | 5 |
| Learning rate | 1e-4 |
| Warmup | 3%, 193 updates |
| Micro batch / accumulation | 1 / 32 |
| Effective batch | 32 |
| Precision | bf16 model path, FP32 fusion gate |
| Train / validation records | 41,107 / 5,140 |
| Test | not run |

The saved checkpoints map to epoch 1 through 5 as 1285, 2570, 3855, 5140,
and 6425.

## Validation Protocol

The archived Text-SFT base metrics were produced with batch size 10. Evaluation
batch size changes beam ranking for this model family: a 200-record Fusion
smoke comparison between batch 1 and batch 8 changed Top-1 despite preserving
the same examples and deterministic beam settings. Batch size is therefore a
frozen part of this comparison. All results below use batch size 10.

The Text-SFT base result is Top-1 62.2763%, Top-3 74.0272%, and character
accuracy 97.6520%.

| Epoch | Checkpoint | Top-1 | Top-3 | Character accuracy | Top-3 delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| Base | Text-SFT 5140 | 62.2763% | 74.0272% | 97.6520% | - |
| 1 | 1285 | 61.6537% | 73.4047% | 97.5406% | -0.6226 pp |
| 2 | 2570 | 61.6148% | 73.9300% | 97.5457% | -0.0973 pp |
| 3 | 3855 | 61.7315% | 73.5214% | 97.5431% | -0.5058 pp |
| 4 | 5140 | 61.6342% | 73.8521% | 97.5237% | -0.1751 pp |
| 5 | 6425 | 61.6926% | 73.5798% | 97.5360% | -0.4475 pp |

All five Fusion prediction files contain 5,140 validation records. Epoch 2
has the strongest Fusion Top-3 result, but its Top-1 is 0.6615 percentage
points below the base. Epoch 3 has the strongest Fusion Top-1 result, but it
is still 0.5447 percentage points below the base. No checkpoint improves both
selection metrics, and no epoch trend supports continuing longer.

At epoch 2, phonetic Top-3 equals the base phonetic Top-3 at 71.0287%, while
identity Top-3 falls from the base 95.1487% to 94.3662%. The fusion residual
therefore preserves some phonetic candidate coverage but harms ranking and
identity behavior enough to lower the all-record result.

## Artifact Boundaries

Checkpoints, predictions, logs, and other large artifacts remain outside Git.
Git tracks source, runbooks, and small reproducibility code only. The local
patch backup named evaluate_adapter.py.orig is deliberately not part of this
experiment record.

Explicit IPA completed its small overfit gate but has no corresponding full
validation result in this report. It is not a comparator for the Fusion
decision above.
