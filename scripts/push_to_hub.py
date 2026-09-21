#!/usr/bin/env python
"""Generate model cards for the four delivered models and push them to the HuggingFace Hub.

    python scripts/push_to_hub.py --user <hf-username> --dry-run   # write cards, push nothing
    python scripts/push_to_hub.py --user <hf-username>             # create repos and upload
    python scripts/push_to_hub.py --user <hf-username> --private   # ...as private repos

Authentication: run `hf auth login` (or `huggingface-cli login`) once, or export HF_TOKEN.
Which run is delivered per task comes from configs/delivery.json, falling back to the best
headline metric in results/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import MODELS, ROOT as SRC_ROOT  # noqa: E402
from src.report_data import HEADLINE, SECONDARY, TASK_TITLES, best_per_task, rows  # noqa: E402

DELIVERY = ROOT / "configs" / "delivery.json"
CARDS = ROOT / "docs" / "model_cards"

REPO_SLUG = {"agnews": "bert-base-uncased-agnews-topic",
             "ner": "bert-base-uncased-conll2003-ner",
             "pos": "bert-base-uncased-ud-ewt-pos",
             "qa": "bert-base-uncased-squad-qa"}

PIPELINE = {"agnews": "text-classification", "ner": "token-classification",
            "pos": "token-classification", "qa": "question-answering"}

HF_DATASET = {"agnews": "fancyzhx/ag_news", "ner": "lhoestq/conll2003",
              "pos": "universal-dependencies/universal_dependencies",
              "qa": "rajpurkar/squad"}

INTENDED = {
    "agnews": "Assigning one of four news topics (World, Sports, Business, Sci/Tech) to a short "
              "English news headline plus lead sentence.",
    "ner": "Tagging English newswire text with PER / ORG / LOC / MISC entity spans in BIO format.",
    "pos": "Tagging English text with the 17 Universal Dependencies part-of-speech tags.",
    "qa": "Extracting the answer span to a question from a paragraph that is known to contain it "
          "(SQuAD v1.1 style; the model always returns a span and cannot abstain).",
}

LIMITS = {
    "agnews": "Trained on 2004-era news wire, so topic drift is real: modern entities and "
              "technology vocabulary are under-represented, and the Business / Sci-Tech boundary "
              "is the model's weakest. Texts far longer than a headline and lead are truncated at "
              "128 subwords. English only.",
    "ner": "CoNLL-2003 is Reuters newswire from 1996-1997. The MISC class is heterogeneous and the "
           "weakest; entity types outside PER/ORG/LOC/MISC are not modelled; lowercase or noisy "
           "user-generated text degrades sharply because casing carries much of the signal. "
           "English only.",
    "pos": "UD English-EWT is web text (blogs, e-mail, reviews, newsgroups); performance on other "
           "domains and on languages other than English is not measured here. Tags are the 17 "
           "coarse UPOS categories, not the finer XPOS set.",
    "qa": "Trained on a 15 000-example subsample of SQuAD v1.1, so it is below published "
          "full-data numbers. It assumes the answer is present in the context: on an unanswerable "
          "question it still returns its best span. Contexts longer than 384 subwords are "
          "processed in overlapping windows. English only.",
}


def resolve_delivery() -> dict[str, str]:
    best = {t: r["run_id"] for t, r in best_per_task().items()}
    if DELIVERY.exists():
        best.update(json.loads(DELIVERY.read_text()))
    return best


def card(row: dict, repo: str, task: str) -> str:
    rec, m = row["raw"], row["raw"]["metrics"]
    hp, params, ds = rec["hyperparams"], rec["params"], rec["dataset"]
    hkey, hname = HEADLINE[task]
    skey, sname = SECONDARY[task]
    pct = (lambda v: f"{v * 100:.2f}") if task != "qa" else (lambda v: f"{v:.2f}")
    metric_lines = "\n".join(
        f"      - type: {k}\n        value: {pct(m[k])}\n        name: {n}"
        for k, n in ((hkey, hname), (skey, sname)) if k in m)

    comparison = "\n".join(
        f"| {r['method']} | {r['trainable']:,} | {r['trainable_pct']}% | "
        f"{r['headline']:.2f} | {r['secondary']:.2f} | {r['minutes']:.1f} |"
        for r in rows(task) if r["headline"] is not None)

    labels = ds.get("labels") or []
    label_block = ("\n".join(f"- `{i}` `{l}`" for i, l in enumerate(labels))
                   if labels else "Start / end position logits over the context tokens.")

    return f"""---
language: en
license: apache-2.0
base_model: {rec['model_name']}
pipeline_tag: {PIPELINE[task]}
library_name: transformers
tags:
  - bert
  - {PIPELINE[task]}
  - {task}
datasets:
  - {HF_DATASET[task]}
metrics:
  - {hkey}
  - {skey}
model-index:
  - name: {repo}
    results:
    - task:
        type: {PIPELINE[task]}
        name: {TASK_TITLES[task]}
      dataset:
        name: {ds['source']}
        type: {HF_DATASET[task]}
      metrics:
{metric_lines}
---

# {repo}

`{rec['model_name']}` adapted to **{TASK_TITLES[task]}** with **{row['method']}**.

Produced for the assignment *U2T01 - Adapting BERT for NLP tasks* (Trends in Data Science).
The delivered method was chosen by measurement, not by default: the table below is the full
set of adaptation methods trained for this task, and this repository holds the winner.

## Intended use

{INTENDED[task]}

Out of scope: any use where an error carries real cost without a human in the loop, and any
language or domain other than the one above.

## Training data

- **Dataset:** `{ds['source']}`
- **License:** {ds['license']}
- **Citation:** {ds['citation']}
- **Splits used:** {json.dumps(ds['sizes'])}

## Adaptation method

| | |
|---|---|
| Method | {row['method']} |
| Trainable parameters | {params.get('trainable_params', 0):,} of {params.get('total_params', 0):,} ({params.get('trainable_pct', 0)}%) |
| Head learning rate | {hp.get('head_lr')} |
| Encoder learning rate | {hp.get('body_lr')} |
| Epochs / batch size | {hp.get('epochs')} / {hp.get('batch_size')} |
| Max sequence length | {hp.get('max_length')} |
| Scheduler | {hp.get('scheduler', 'linear with 10% warmup')} |
| Seed | {rec['seed']} |
| Hardware | {rec['env'].get('chip', rec['env'].get('device'))} ({rec['env'].get('device')}) |
| Wall-clock training time | {row['minutes']:.1f} min |

A freshly initialised head and pretrained encoder weights are trained in **two parameter
groups with separate learning rates**; a single shared rate either starves the head or
destroys pretrained features.

## Evaluation

Held-out test split, never seen during training or model selection.

| Metric | Value |
|---|---|
| **{hname}** | **{pct(m[hkey])}** |
| {sname} | {pct(m[skey])} |

### What every method scored on this task

| Method | Trainable params | Share of model | {hname} | {sname} | Train time (min) |
|---|---:|---:|---:|---:|---:|
{comparison}

Single run per configuration with a fixed seed. Re-running with a different seed moves these
numbers by roughly +/- 1-3 points, so gaps smaller than that are noise rather than findings.

## Labels

{label_block}

## How to use

```python
from transformers import pipeline

pipe = pipeline("{PIPELINE[task]}", model="{repo}")
```

## Limitations and bias

{LIMITS[task]}

The model inherits whatever social bias is present in BERT's pretraining corpus
(BooksCorpus + English Wikipedia) and in the task dataset above.

## References

- Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) - section 5.3
  is the feature-based vs fine-tuning comparison this work mirrors.
- {ds['citation']}
- HuggingFace Transformers, [fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="HuggingFace username or org")
    ap.add_argument("--dry-run", action="store_true", help="write cards, do not touch the Hub")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=["agnews", "ner", "pos", "qa"])
    args = ap.parse_args()

    delivery = resolve_delivery()
    CARDS.mkdir(parents=True, exist_ok=True)
    by_id = {(r["task"], r["run_id"]): r for r in rows()}

    pushed = []
    for task in args.tasks:
        run_id = delivery.get(task)
        row = by_id.get((task, run_id))
        if row is None:
            print(f"[skip] {task}: no result for run_id {run_id!r}", file=sys.stderr)
            continue
        repo = f"{args.user}/{REPO_SLUG[task]}"
        text = card(row, repo, task)
        card_path = CARDS / f"{REPO_SLUG[task]}.md"
        card_path.write_text(text)
        local = MODELS / task / run_id
        print(f"[card] {card_path.relative_to(ROOT)}  <- {task}/{run_id}")

        if args.dry_run:
            continue
        if not local.exists():
            print(f"[error] {local} missing - rerun that experiment with save_model=True",
                  file=sys.stderr)
            continue
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, exist_ok=True, private=args.private)
        (local / "README.md").write_text(text)
        api.upload_folder(folder_path=str(local), repo_id=repo,
                          commit_message=f"U2T01: {row['method']} for {TASK_TITLES[task]}")
        pushed.append(repo)
        print(f"[push] https://huggingface.co/{repo}")

    if args.dry_run:
        print(f"\n[dry-run] {len(args.tasks)} model cards written to {CARDS.relative_to(ROOT)}; "
              "nothing was uploaded.")
    else:
        print(f"\n[done] {len(pushed)} repo(s) on the Hub: {', '.join(pushed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
