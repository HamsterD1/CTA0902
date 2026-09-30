# Fusion Epoch-2 Paired Diagnosis

Status: validation-only diagnostic, 2026-09-30. No model training, test
evaluation, extra epochs, or extra seeds were run.

## Scope and comparability

The archived Text-SFT benchmark remains Top-1 62.2763%, Top-3 74.0272%, and
character accuracy 97.6520%. Its original per-record prediction file is not
available on the migrated host, so it cannot support an exact historical
paired comparison. A direct sequential batch-10 re-evaluation also drifted
(62.3152% / 73.8132%), showing that batch layout affects deterministic beam
ranking for this linear-attention model.

For a valid paired counterfactual, Text-SFT checkpoint-5140 was regenerated on
the same 5,140 validation rows with beam 3, batch cap 10, same prompt,
tokenizer, max-new-tokens 384, and the same prompt-length cohort layout used
by the Fusion evaluator. This companion result is 62.1984% / 73.8716% /
97.6602%. It is used only to diagnose the already archived Fusion epoch-2
predictions, never to replace the frozen benchmark.

## Paired result

| Metric | Text-SFT paired control | Fusion epoch-2 | Delta |
| --- | ---: | ---: | ---: |
| Top-1 | 62.1984% | 61.6148% | -0.5837 pp |
| Top-3 | 73.8716% | 73.9300% | +0.0584 pp |
| Char-Acc | 97.6602% | 97.5457% | -0.1146 pp |

Identity loses Top-1 1.2520 pp, Top-3 0.4695 pp, and character accuracy
0.3715 pp. Phonetic Top-3 gains 0.1333 pp but loses Top-1 0.4888 pp and
character accuracy 0.0810 pp. The available source schema has only
`identity`/`phonetic` and a 17-record `russian` tag; it contains no finer
variant-type label.

## Candidate movement

- Base Top-1 correct -> Fusion wrong: 97; Fusion retains gold at rank 2 for
  63 and rank 3 for 11, but removes it from Top-3 for 23.
- Base Top-1 wrong -> Fusion correct: 67; gold was already at rank 2/3 for 60
  and absent for 7.
- Thus 74/97 (76.3%) losses are candidate demotions, while the net Top-1 loss
  is 30 records. Candidate coverage also changes: 2,026 records have a
  different candidate set.
- Losses cluster in 33-48 text-token prompts (52/97) and short-to-medium IPA:
  5-8 phonemes loses 2.5455 Top-1 pp and 1.8182 Top-3 pp. Long IPA (>=17)
  gains 0.4091 Top-3 pp but still loses 0.1259 Top-1 pp.

The stored evaluators do not retain generation scores or logits, so this
diagnosis compares full candidate sequences and gold ranks rather than
post-hoc logits.

## Decision

The evidence supports a minimal next experiment that preserves the frozen
Text-SFT main path and weakens or gates the residual Fusion injection. Do not
continue the current Fusion configuration, add epochs, run seeds 43/44, or
run test. This is stronger than a conclusion that XPhoneBERT is wholly
uninformative: phonetic Top-3 candidate coverage has a small positive signal,
but it is currently outweighed by candidate re-ranking and identity damage.
