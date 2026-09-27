# Cross-hardware replica: Apple M5 vs Colab T4

A teammate ran the complete grid a second time on a Colab T4 through
`notebooks/U2T01_colab.ipynb`, from the same code and the same seed, on 2026-09-25. Every run
that also exists locally is compared below on its task's primary metric.

The local runs stay the source of truth for `results/`: they are the ones whose checkpoints
are published, so the Hub models and the report agree to the last decimal. The one run adopted
from the T4 is **SQuAD full fine-tuning**, which never completed locally (see section 6.4 of the
report); it has no local twin and does not appear in this table.

| Task | Method | Metric | Apple M5 (mps, bf16) | Colab T4 (cuda, fp16) | Δ |
|---|---|---|---:|---:|---:|
| Topic classification | Feature-based (logistic regression, mean-pooled) | Accuracy | 90.18 | 90.14 | -0.04 |
| Topic classification | Feature-based (linear probe, pooler as head) | Accuracy | 89.61 | 89.62 | +0.01 |
| Topic classification | Feature-based (linear SVM) | Accuracy | 89.21 | 89.17 | -0.04 |
| Topic classification | Feature-based (logistic regression) | Accuracy | 88.61 | 88.57 | -0.04 |
| Topic classification | Feature-based (random forest) | Accuracy | 84.87 | 84.72 | -0.14 |
| Topic classification | Feature-based (MLP probe) | Accuracy | 84.58 | 84.62 | +0.04 |
| Topic classification | Feature-based (linear probe) | Accuracy | 82.09 | 82.13 | +0.04 |
| Topic classification | Partial FT (top 2 layers) | Accuracy | 92.36 | 92.34 | -0.01 |
| Topic classification | Full fine-tuning | Accuracy | 93.01 | 92.93 | -0.08 |
| Named entity recognition | Feature-based (MLP probe) | Entity F1 | 84.17 | 84.04 | -0.13 |
| Named entity recognition | Feature-based (linear probe) | Entity F1 | 80.90 | 80.73 | -0.17 |
| Named entity recognition | Feature-based (logistic regression) | Entity F1 | 77.84 | 78.00 | +0.15 |
| Named entity recognition | Partial FT (top 4 layers) | Entity F1 | 88.10 | 88.03 | -0.07 |
| Named entity recognition | Full fine-tuning | Entity F1 | 90.41 | 90.11 | -0.30 |
| Part-of-speech tagging | Feature-based (logistic regression) | Token accuracy | 93.94 | 93.95 | +0.01 |
| Part-of-speech tagging | Feature-based (linear probe) | Token accuracy | 93.27 | 93.25 | -0.02 |
| Part-of-speech tagging | Partial FT (top 2 layers) | Token accuracy | 95.66 | 95.63 | -0.02 |
| Part-of-speech tagging | Full fine-tuning | Token accuracy | 97.51 | 97.44 | -0.07 |
| Extractive QA | Feature-based (linear probe) | F1 | 25.18 | 25.21 | +0.03 |
| Extractive QA | Partial FT (top 4 layers) | F1 | 72.91 | 72.89 | -0.02 |

**20 runs, largest difference 0.30 points, mean 0.07.**
Different device, different reduced-precision format (bfloat16 on MPS, float16 on the T4, which
has no bfloat16), different torch build — and no run moves by more than a third of a point, an
order of magnitude inside the ±1–3 point seed band. Two consequences:

- A replica on other hardware should land within a few tenths of every number in the report.
- Mixing the one T4 run into the SQuAD comparison is safe for the metric: the T4 re-scored
  partial fine-tuning within 0.02 F1 of the local run, against a 9.2-point gap between partial and
  full fine-tuning. Wall-clock times are **not** comparable across the two machines, and the
  report marks the T4 row with † wherever a time is shown.

## The adopted checkpoint, verified on both machines

The notebook also reloaded every checkpoint it had saved and re-scored it on the T4: **15/15
reproduced their metric exactly**, including SQuAD full fine-tuning at 82.137 F1 (Δ 0.0000).
That is the checkpoint published as `joseeangel/bert-base-uncased-squad-qa`; its
`model.safetensors` on the Hub has the same SHA-256 as the local copy
(`af11e341…ad35d81e`).

Re-scored on the Apple M5 instead, the same weights give 82.173 F1, a difference of 0.036
points. Nothing changed but the evaluation device and with it the reduced-precision format
(float16 on the T4, bfloat16 on MPS), which is enough to flip a handful of near-tied answer
spans out of 10 570 questions. `docs/model_checks.md` records it as a cross-device row.
