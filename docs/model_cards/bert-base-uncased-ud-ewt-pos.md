---
language: en
license: apache-2.0
base_model: bert-base-uncased
pipeline_tag: token-classification
library_name: transformers
tags:
  - bert
  - token-classification
  - pos
datasets:
  - universal-dependencies/universal_dependencies
metrics:
  - token_accuracy
  - macro_f1
model-index:
  - name: joseeangel/bert-base-uncased-ud-ewt-pos
    results:
    - task:
        type: token-classification
        name: Part-of-speech tagging (UD English-EWT)
      dataset:
        name: universal-dependencies/universal_dependencies (config en_ewt)
        type: universal-dependencies/universal_dependencies
      metrics:
      - type: token_accuracy
        value: 97.51
        name: Token accuracy
      - type: macro_f1
        value: 93.70
        name: Macro F1
---

# joseeangel/bert-base-uncased-ud-ewt-pos

`bert-base-uncased` adapted to **Part-of-speech tagging (UD English-EWT)** with **Full fine-tuning**.

Produced for the assignment *U2T01 - Adapting BERT for NLP tasks* (Trends in Data Science).
The delivered method was chosen by measurement, not by default: the table below is the full
set of adaptation methods trained for this task, and this repository holds the winner.

## Intended use

Tagging English text with the 17 Universal Dependencies part-of-speech tags.

Out of scope: any use where an error carries real cost without a human in the loop, and any
language or domain other than the one above.

## Training data

- **Dataset:** `universal-dependencies/universal_dependencies (config en_ewt)`
- **License:** CC BY-SA 4.0
- **Citation:** Silveira et al. (2014) / Universal Dependencies 2.x, UD_English-EWT
- **Splits used:** {"train": 12544, "dev": 2001, "test": 2077}

## Adaptation method

| | |
|---|---|
| Method | Full fine-tuning |
| Trainable parameters | 108,904,721 of 108,904,721 (100.0%) |
| Head learning rate | 0.001 |
| Encoder learning rate | 2e-05 |
| Epochs / batch size | 3 / 32 |
| Max sequence length | 192 |
| Scheduler | linear with 10% warmup |
| Seed | 42 |
| Hardware | Apple M5 (mps) |
| Wall-clock training time | 11.0 min |

A freshly initialised head and pretrained encoder weights are trained in **two parameter
groups with separate learning rates**; a single shared rate either starves the head or
destroys pretrained features.

## Evaluation

Held-out test split, never seen during training or model selection.

| Metric | Value |
|---|---|
| **Token accuracy** | **97.51** |
| Macro F1 | 93.70 |

### What every method scored on this task

| Method | Trainable params | Share of model | Token accuracy | Macro F1 | Train time (min) |
|---|---:|---:|---:|---:|---:|
| Feature-based (logistic regression) | 13,073 | 0.0% | 93.94 | 88.72 | 1.6 |
| Feature-based (linear probe) | 13,073 | 0.012% | 93.27 | 86.77 | 4.1 |
| Partial FT (top 2 layers) | 14,188,817 | 13.029% | 95.66 | 89.06 | 3.6 |
| Full fine-tuning | 108,904,721 | 100.0% | 97.51 | 93.70 | 11.0 |

Single run per configuration with a fixed seed. Re-running with a different seed moves these
numbers by roughly +/- 1-3 points, so gaps smaller than that are noise rather than findings.

## Labels

- `0` `ADJ`
- `1` `ADP`
- `2` `ADV`
- `3` `AUX`
- `4` `CCONJ`
- `5` `DET`
- `6` `INTJ`
- `7` `NOUN`
- `8` `NUM`
- `9` `PART`
- `10` `PRON`
- `11` `PROPN`
- `12` `PUNCT`
- `13` `SCONJ`
- `14` `SYM`
- `15` `VERB`
- `16` `X`

## How to use

```python
from transformers import pipeline

pipe = pipeline("token-classification", model="joseeangel/bert-base-uncased-ud-ewt-pos")
```

## Limitations and bias

UD English-EWT is web text (blogs, e-mail, reviews, newsgroups); performance on other domains and on languages other than English is not measured here. Tags are the 17 coarse UPOS categories, not the finer XPOS set.

The model inherits whatever social bias is present in BERT's pretraining corpus
(BooksCorpus + English Wikipedia) and in the task dataset above.

## References

- Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) - section 5.3
  is the feature-based vs fine-tuning comparison this work mirrors.
- Silveira et al. (2014) / Universal Dependencies 2.x, UD_English-EWT
- HuggingFace Transformers, [fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).
