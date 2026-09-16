# Phonetic Variant Restoration

This recipe implements Section 2.2 of `docx/拟音变体还原实验.md`: three full
fine-tuning mT5-base ablations over structured Seq2Seq triples:

```text
(variant_text, ipa, canonical_text)
```

## Layout

```text
data/variant-restoration/mvp-v1/
  seq2seq-v1/{train,validation,test}.jsonl  # model-facing triples
  manifest.json                             # provenance and split contract
egs/variant-restoration/
  train.py                                  # one condition, one run
  run_mvp_h100.sh                           # all three H100 runs
  evaluate.py                               # EM, character accuracy, CER
```

The data split has 1,009 train, 158 validation, and 185 test records. Test
records are *Seen Canonical + Unseen Variant*: their canonical target is in the
training set, but the precise variant-target pair is not.

## H100 Setup

On the remote H100 host, create an isolated Python environment and install the
recipe dependencies. Use a CUDA-compatible PyTorch build supplied by the host
image or install one appropriate to that host before installing the rest.

```bash
python -m pip install -r egs/variant-restoration/requirements-h100.txt
```

Run all ablations from the repository root:

```bash
bash egs/variant-restoration/run_mvp_h100.sh
```

Outputs are placed under `exp/variant-restoration/mvp-v1/{text,ipa,text_ipa}`.
Each run saves its best checkpoint, tokenizer, Trainer state, `test_results.json`,
decoded `test_predictions.jsonl`, and `test_report.json`.

## H100 Defaults

The launcher uses mT5-base full fine-tuning with BF16, TF32, gradient
checkpointing, batch size 16, and accumulation 2 (effective batch size 32).
The model selection metric is validation exact-match accuracy. All random seeds
are fixed to 42. Change only one setting per follow-up run and preserve the
output directory of the baseline run.

## v3 IPA-to-Text Training

The retained v3 combined export is the input for the current phoneme-to-text
run. It contains 44,824 `phonetic` records and 6,928 `identity` records.
`sample_type` is preserved throughout data preparation and evaluation.

```bash
python -m pip install -r egs/variant-restoration/requirements-h100.txt
mkdir -p exp/variant-restoration/v3-mt5-ipa
bash egs/variant-restoration/run_v3_ipa_mt5_h100.sh
```

The launcher keeps each `canonical_text` in one split only, removes exact
duplicate triples, and samples the training identity records to 10% by default.
Override that ratio deliberately, for example:

```bash
IDENTITY_SAMPLE_RATIO=0.15 CHECKPOINT_ROOT=/cpt_dlt/variant-restoration/v3-mt5-ipa-r15 \
  bash egs/variant-restoration/run_v3_ipa_mt5_h100.sh
```

Prepared JSONL splits and checkpoints remain under `CHECKPOINT_ROOT`. Concise
reports live in `exp/variant-restoration/v3-mt5-ipa/` and are ignored by Git.
The final report includes `all`, `phonetic`, and `identity` buckets.
