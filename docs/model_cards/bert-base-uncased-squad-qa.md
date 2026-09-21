---
language: en
license: apache-2.0
base_model: bert-base-uncased
pipeline_tag: question-answering
library_name: transformers
tags:
  - bert
  - question-answering
  - qa
datasets:
  - rajpurkar/squad
metrics:
  - f1
  - exact_match
model-index:
  - name: joseeangel/bert-base-uncased-squad-qa
    results:
    - task:
        type: question-answering
        name: Extractive QA (SQuAD v1.1)
      dataset:
        name: rajpurkar/squad (v1.1)
        type: rajpurkar/squad
      metrics:
      - type: f1
        value: 72.91
        name: F1
      - type: exact_match
        value: 60.68
        name: Exact match
---

# joseeangel/bert-base-uncased-squad-qa

`bert-base-uncased` adapted to **Extractive QA (SQuAD v1.1)** with **Partial FT (top 4 layers)**.

Produced for the assignment *U2T01 - Adapting BERT for NLP tasks* (Trends in Data Science).
The delivered method was chosen by measurement, not by default: the table below is the full
set of adaptation methods trained for this task, and this repository holds the winner.

## Intended use

Extracting the answer span to a question from a paragraph that is known to contain it (SQuAD v1.1 style; the model always returns a span and cannot abstain).

Out of scope: any use where an error carries real cost without a human in the loop, and any
language or domain other than the one above.

## Training data

- **Dataset:** `rajpurkar/squad (v1.1)`
- **License:** CC BY-SA 4.0
- **Citation:** Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text
- **Splits used:** {"train": 15000, "dev": 3000, "test": 10570}

## Adaptation method

| | |
|---|---|
| Method | Partial FT (top 4 layers) |
| Trainable parameters | 28,353,026 of 108,893,186 (26.037%) |
| Head learning rate | 0.001 |
| Encoder learning rate | 3e-05 |
| Epochs / batch size | 2 / 16 |
| Max sequence length | 384 |
| Scheduler | linear with 10% warmup |
| Seed | 42 |
| Hardware | Apple M5 (mps) |
| Wall-clock training time | 25.7 min |

A freshly initialised head and pretrained encoder weights are trained in **two parameter
groups with separate learning rates**; a single shared rate either starves the head or
destroys pretrained features.

## Evaluation

Held-out test split, never seen during training or model selection.

| Metric | Value |
|---|---|
| **F1** | **72.91** |
| Exact match | 60.68 |

### What every method scored on this task

| Method | Trainable params | Share of model | F1 | Exact match | Train time (min) |
|---|---:|---:|---:|---:|---:|
| Feature-based (linear probe) | 1,538 | 0.001% | 25.18 | 16.20 | 25.0 |
| Partial FT (top 4 layers) | 28,353,026 | 26.037% | 72.91 | 60.68 | 25.7 |

Single run per configuration with a fixed seed. Re-running with a different seed moves these
numbers by roughly +/- 1-3 points, so gaps smaller than that are noise rather than findings.

## Labels

Start / end position logits over the context tokens.

## How to use

```python
from transformers import pipeline

pipe = pipeline("question-answering", model="joseeangel/bert-base-uncased-squad-qa")
```

## Limitations and bias

Trained on a 15 000-example subsample of SQuAD v1.1, so it is below published full-data numbers. It assumes the answer is present in the context: on an unanswerable question it still returns its best span. Contexts longer than 384 subwords are processed in overlapping windows. English only.

The model inherits whatever social bias is present in BERT's pretraining corpus
(BooksCorpus + English Wikipedia) and in the task dataset above.

## References

- Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) - section 5.3
  is the feature-based vs fine-tuning comparison this work mirrors.
- Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text
- HuggingFace Transformers, [fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).
