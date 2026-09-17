# Variant Restoration Experiments

This directory contains results for two different experimental scales. The
directory names below are the authoritative mapping; do not infer the scale
from an incidental `v1`, `v3`, or model suffix in a historical path.

| Directory | Experiment | Data | Status |
| --- | --- | --- | --- |
| `mvp-v1/` | Small-data MVP | 1,352 structured Seq2Seq triples: 1,009 train, 158 validation, 185 test | Historical MVP baseline. The test contract is seen canonical target with unseen exact variant-target pair. |
| `v3-mt5-ipa/` | Full-data first baseline | 51,752 combined v3 records before exact deduplication; 51,389 unique triples after removing 363 duplicates | Completed. This historical folder name is retained to preserve the published result links. It is not the small MVP. Its test contract holds out all `canonical_text` targets. |
| `full-data-mt5/<RUN_NAME>/` | Full-data follow-up runs | The same retained combined v3 training export, with a descriptive run name such as `long-context-b12` | Current location for new full-data mT5 IPA-to-text runs. Every run writes a token-length audit and its resolved training configuration. |

## Completed Full-Data Baseline

`v3-mt5-ipa/` is the first full-data mT5 IPA-only baseline, not an experiment
version to extend. It trained on 44,826 sampled training records after grouped
splitting and 10% identity sampling, for five epochs with 512/256 token limits
and effective batch size 32. Its result commit is `71e2293`; the analysis is at
`v3-mt5-ipa/analysis-2026-09-16/README.md`.

## New Full-Data Runs

Use `egs/variant-restoration/run_ipa_mt5_h100.sh` with a descriptive `RUN_NAME`.
The default path is `full-data-mt5/<RUN_NAME>/`, so a run name records the
experiment's purpose rather than a sequence number. Example:

```bash
RUN_NAME=long-context-b12 \
MODEL_NAME=/model_dlt/mt5-base \
bash egs/variant-restoration/run_ipa_mt5_h100.sh
```

This does not overwrite `mvp-v1/` or the completed full-data baseline.
