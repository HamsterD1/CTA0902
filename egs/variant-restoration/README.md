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
