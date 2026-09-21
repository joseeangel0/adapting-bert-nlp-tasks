---
language: en
license: apache-2.0
base_model: bert-base-uncased
pipeline_tag: token-classification
library_name: transformers
tags:
  - bert
  - token-classification
  - ner
datasets:
  - lhoestq/conll2003
metrics:
  - entity_f1
  - token_accuracy
model-index:
  - name: joseeangel/bert-base-uncased-conll2003-ner
    results:
    - task:
        type: token-classification
        name: Named entity recognition (CoNLL-2003)
      dataset:
        name: lhoestq/conll2003
        type: lhoestq/conll2003
      metrics:
      - type: entity_f1
        value: 90.41
        name: Entity F1
      - type: token_accuracy
        value: 98.08
        name: Token accuracy
---

# joseeangel/bert-base-uncased-conll2003-ner

`bert-base-uncased` adapted to **Named entity recognition (CoNLL-2003)** with **Full fine-tuning**.

Produced for the assignment *U2T01 - Adapting BERT for NLP tasks* (Trends in Data Science).
The delivered method was chosen by measurement, not by default: the table below is the full
set of adaptation methods trained for this task, and this repository holds the winner.

## Intended use

Tagging English newswire text with PER / ORG / LOC / MISC entity spans in BIO format.

Out of scope: any use where an error carries real cost without a human in the loop, and any
language or domain other than the one above.

## Training data

- **Dataset:** `lhoestq/conll2003`
- **License:** Reuters corpus terms / research use (CoNLL-2003 shared task)
- **Citation:** Tjong Kim Sang & De Meulder (2003), CoNLL-2003 shared task: language-independent NER
- **Splits used:** {"train": 14041, "dev": 3250, "test": 3453}

## Adaptation method

| | |
|---|---|
| Method | Full fine-tuning |
| Trainable parameters | 108,898,569 of 108,898,569 (100.0%) |
| Head learning rate | 0.001 |
| Encoder learning rate | 2e-05 |
| Epochs / batch size | 3 / 32 |
| Max sequence length | 192 |
| Scheduler | linear with 10% warmup |
| Seed | 42 |
| Hardware | Apple M5 (mps) |
| Wall-clock training time | 10.6 min |

A freshly initialised head and pretrained encoder weights are trained in **two parameter
groups with separate learning rates**; a single shared rate either starves the head or
destroys pretrained features.

## Evaluation

Held-out test split, never seen during training or model selection.

| Metric | Value |
|---|---|
| **Entity F1** | **90.41** |
| Token accuracy | 98.08 |

### What every method scored on this task

| Method | Trainable params | Share of model | Entity F1 | Token accuracy | Train time (min) |
|---|---:|---:|---:|---:|---:|
| Feature-based (MLP probe) | 398,345 | 0.364% | 84.17 | 97.25 | 4.2 |
| Feature-based (linear probe) | 6,921 | 0.006% | 80.90 | 96.58 | 4.3 |
| Feature-based (logistic regression) | 6,921 | 0.0% | 77.84 | 96.10 | 1.2 |
| Partial FT (top 4 layers) | 28,358,409 | 26.041% | 88.10 | 97.71 | 4.6 |
| Full fine-tuning | 108,898,569 | 100.0% | 90.41 | 98.08 | 10.6 |

Single run per configuration with a fixed seed. Re-running with a different seed moves these
numbers by roughly +/- 1-3 points, so gaps smaller than that are noise rather than findings.

## Labels

- `0` `O`
- `1` `B-PER`
- `2` `I-PER`
- `3` `B-ORG`
- `4` `I-ORG`
- `5` `B-LOC`
- `6` `I-LOC`
- `7` `B-MISC`
- `8` `I-MISC`

## How to use

```python
from transformers import pipeline

pipe = pipeline("token-classification", model="joseeangel/bert-base-uncased-conll2003-ner")
```

## Limitations and bias

CoNLL-2003 is Reuters newswire from 1996-1997. The MISC class is heterogeneous and the weakest; entity types outside PER/ORG/LOC/MISC are not modelled; lowercase or noisy user-generated text degrades sharply because casing carries much of the signal. English only.

The model inherits whatever social bias is present in BERT's pretraining corpus
(BooksCorpus + English Wikipedia) and in the task dataset above.

## References

- Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) - section 5.3
  is the feature-based vs fine-tuning comparison this work mirrors.
- Tjong Kim Sang & De Meulder (2003), CoNLL-2003 shared task: language-independent NER
- HuggingFace Transformers, [fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).
