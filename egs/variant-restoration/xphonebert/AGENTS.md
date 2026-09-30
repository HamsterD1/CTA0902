# XPhoneBERT Experiment Rules

Scope: egs/variant-restoration/xphonebert.

- Treat the v3 IPA, split, prompt/hash, normalization, metric rules, Text-SFT
  checkpoint-5140, and XPhoneBERT revision cf2bc63858dec1c03880fa8f764fe2195accb1ab
  as immutable unless the experiment owner explicitly changes the protocol.
- Keep Qwen and XPhoneBERT frozen for Fusion. Do not silently switch to LoRA,
  full fine-tuning, truncation, or a different IPA windowing policy.
- Validation uses deterministic beam-3 with three returned hypotheses and no
  sampling. The historical Text-SFT comparison uses batch size 10; treat batch
  size as protocol-bearing because it can change beam ranking.
- Never run the test split until the owner selects one validation checkpoint.
  Do not start seeds 43/44 unless seed 42 satisfies the documented stage gate.
- Store checkpoints, predictions, and logs outside Git. Do not stage weights,
  data, secrets, or local patch backups. Preserve unrelated dirty files.
- Read README.md, docs/xphonebert-fusion-warmstart-seed42.md, and the current
  follow-up record docs/xphonebert-fusion-weak-residual-cap01-seed42.md before
  modifying this recipe. The historical Fusion warm-start is learnable but did
  not beat the frozen Text-SFT base. A single bounded weak-residual follow-up
  is validation-only and uses a documented mixed-batch continuation; do not
  treat it as an interchangeable five-epoch replication.
