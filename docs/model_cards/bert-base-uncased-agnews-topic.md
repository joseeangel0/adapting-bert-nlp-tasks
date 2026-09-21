---
language: en
license: apache-2.0
base_model: bert-base-uncased
pipeline_tag: text-classification
library_name: transformers
tags:
  - bert
  - text-classification
  - agnews
datasets:
  - fancyzhx/ag_news
metrics:
  - accuracy
  - macro_f1
model-index:
  - name: joseeangel/bert-base-uncased-agnews-topic
    results:
    - task:
        type: text-classification
        name: Topic classification (AG News)
      dataset:
        name: fancyzhx/ag_news
        type: fancyzhx/ag_news
      metrics:
      - type: accuracy
        value: 93.01
        name: Accuracy
      - type: macro_f1
        value: 93.00
        name: Macro F1
---

# joseeangel/bert-base-uncased-agnews-topic

`bert-base-uncased` adapted to **Topic classification (AG News)** with **Full fine-tuning**.

Produced for the assignment *U2T01 - Adapting BERT for NLP tasks* (Trends in Data Science).
The delivered method was chosen by measurement, not by default: the table below is the full
set of adaptation methods trained for this task, and this repository holds the winner.

## Intended use

Assigning one of four news topics (World, Sports, Business, Sci/Tech) to a short English news headline plus lead sentence.

Out of scope: any use where an error carries real cost without a human in the loop, and any
language or domain other than the one above.

## Training data

- **Dataset:** `fancyzhx/ag_news`
- **License:** Custom (AG News corpus, academic use)
- **Citation:** Zhang, Zhao & LeCun (2015), Character-level Convolutional Networks for Text Classification
- **Splits used:** {"train": 20000, "dev": 5000, "test": 7600}

## Adaptation method

| | |
|---|---|
| Method | Full fine-tuning |
| Trainable parameters | 109,485,316 of 109,485,316 (100.0%) |
| Head learning rate | 0.001 |
| Encoder learning rate | 2e-05 |
| Epochs / batch size | 2 / 32 |
| Max sequence length | 128 |
| Scheduler | linear with 10% warmup |
| Seed | 42 |
| Hardware | Apple M5 (mps) |
| Wall-clock training time | 16.6 min |

A freshly initialised head and pretrained encoder weights are trained in **two parameter
groups with separate learning rates**; a single shared rate either starves the head or
destroys pretrained features.

## Evaluation

Held-out test split, never seen during training or model selection.

| Metric | Value |
|---|---|
| **Accuracy** | **93.01** |
| Macro F1 | 93.00 |

### What every method scored on this task

| Method | Trainable params | Share of model | Accuracy | Macro F1 | Train time (min) |
|---|---:|---:|---:|---:|---:|
| Feature-based (logistic regression, mean-pooled) | 3,076 | 0.0% | 90.18 | 90.18 | 2.9 |
| Feature-based (linear probe, pooler as head) | 593,668 | 0.542% | 89.61 | 89.59 | 13.6 |
| Feature-based (linear SVM) | 3,076 | 0.0% | 89.21 | 89.20 | 0.1 |
| Feature-based (logistic regression) | 3,076 | 0.0% | 88.61 | 88.59 | 2.0 |
| Feature-based (random forest) | 0 | 0.0% | 84.87 | 84.79 | 0.3 |
| Feature-based (MLP probe) | 395,780 | 0.36% | 84.58 | 84.57 | 14.8 |
| Feature-based (linear probe) | 3,076 | 0.003% | 82.09 | 82.04 | 12.5 |
| Partial FT (top 2 layers) | 14,178,820 | 12.95% | 92.36 | 92.35 | 13.1 |
| Full fine-tuning | 109,485,316 | 100.0% | 93.01 | 93.00 | 16.6 |

Single run per configuration with a fixed seed. Re-running with a different seed moves these
numbers by roughly +/- 1-3 points, so gaps smaller than that are noise rather than findings.

## Labels

- `0` `World`
- `1` `Sports`
- `2` `Business`
- `3` `Sci/Tech`

## How to use

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="joseeangel/bert-base-uncased-agnews-topic")
```

## Limitations and bias

Trained on 2004-era news wire, so topic drift is real: modern entities and technology vocabulary are under-represented, and the Business / Sci-Tech boundary is the model's weakest. Texts far longer than a headline and lead are truncated at 128 subwords. English only.

The model inherits whatever social bias is present in BERT's pretraining corpus
(BooksCorpus + English Wikipedia) and in the task dataset above.

## References

- Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) - section 5.3
  is the feature-based vs fine-tuning comparison this work mirrors.
- Zhang, Zhao & LeCun (2015), Character-level Convolutional Networks for Text Classification
- HuggingFace Transformers, [fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).
