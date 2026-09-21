"""Feature-based adaptation: BERT is a frozen feature extractor, a classical model learns.

Zero BERT parameters are updated. The last hidden states are extracted once, cached to
disk, and then consumed by scikit-learn estimators (logistic regression, linear SVM,
random forest). This is the cheapest rung of the ladder and the baseline every
fine-tuned model has to beat to justify its cost.

Pooling: [CLS] for sequence classification (the vector BERT itself pools for NSP), and
the *first-subword* vector for token-level tasks - the same position the -100 convention
puts the label on, so features and labels stay aligned by construction.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

from .common import CACHE, RunRecord, Stopwatch, get_device, set_seed
from .data import TaskData, dataset_card, load_task
from .encoding import IGNORE, MAX_LEN, align_labels
from .metrics import ner_metrics, pos_metrics, sequence_metrics

MAX_TRAIN_TOKENS = 150_000   # cap for token tasks, keeps the sklearn fit in minutes

# Parameter counts of the frozen bodies, so a feature-based run can report what it did *not*
# train without paying to load the weights again.
BODY_PARAMS = {"bert-base-uncased": 109_482_240, "distilbert-base-uncased": 66_362_880,
               "bert-large-uncased": 335_141_888}


@torch.no_grad()
def extract(td: TaskData, split: str, model_name: str, batch_size: int = 64,
            pooling: str = "cls") -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (features, labels, sentence_lengths). Cached on disk by (task, model, split).

    `pooling` applies to sequence classification only. Token-level tasks always take the
    *first-subword* vector of each word - the same position the -100 convention puts the
    label on - so features and labels are aligned by construction.
    """
    pool_tag = pooling if td.kind == "sequence" else "firstsub"
    tag = f"{td.name}_{model_name.split('/')[-1]}_{split}_{pool_tag}_{len(td.splits[split])}"
    path = CACHE / f"{tag}.npz"
    if path.exists():
        z = np.load(path)
        return z["X"].astype(np.float32), z["y"], list(z["lens"])

    CACHE.mkdir(parents=True, exist_ok=True)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    attn = {"attn_implementation": "eager"} if device.type == "mps" else {}
    body = AutoModel.from_pretrained(model_name, **attn).to(device).eval()

    data = td.splits[split]
    feats, labels, lens = [], [], []
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        if td.kind == "sequence":
            enc = tokenizer(batch[td.text_key], truncation=True, max_length=MAX_LEN[td.name],
                            padding=True, return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
                hidden = body(**enc).last_hidden_state.float()
            if pooling == "cls":
                vec = hidden[:, 0]
            else:                                    # attention-masked mean pooling
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1)
            feats.append(vec.cpu().numpy().astype(np.float16))
            labels.extend(batch[td.label_key])
            lens.extend([1] * len(vec))
        else:
            enc = tokenizer(batch[td.text_key], is_split_into_words=True, truncation=True,
                            max_length=MAX_LEN[td.name], padding=True, return_tensors="pt")
            word_ids = [enc.word_ids(i) for i in range(len(batch[td.text_key]))]
            enc_dev = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
                hidden = body(**enc_dev).last_hidden_state.float().cpu().numpy()
            for i, wids in enumerate(word_ids):
                aligned = align_labels(wids, batch[td.label_key][i])
                keep = [k for k, lab in enumerate(aligned) if lab != IGNORE]
                feats.append(hidden[i, keep].astype(np.float16))
                labels.extend(aligned[k] for k in keep)
                lens.append(len(keep))

    X = np.concatenate(feats, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(path, X=X, y=y, lens=np.asarray(lens))
    return X.astype(np.float32), y, lens


def _regroup(flat: np.ndarray, lens: list[int], label_names: list[str]) -> list[list[str]]:
    """Flat per-token array -> list of sentences of tag strings (what seqeval wants)."""
    out, i = [], 0
    for n in lens:
        out.append([label_names[v] for v in flat[i:i + n]])
        i += n
    return out


def build_estimator(kind: str, seed: int = 42):
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=seed)
    if kind == "linsvm":
        return LinearSVC(C=0.5, max_iter=5000, random_state=seed, dual="auto")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                      min_samples_leaf=2)
    raise ValueError(f"unknown estimator {kind}")


def run(task: str, estimator: str = "logreg", model_name: str = "bert-base-uncased", *,
        pooling: str = "cls", seed: int = 42, run_id: str | None = None,
        limit: int | None = None, notes: str = "") -> RunRecord:
    set_seed(seed)
    td = load_task(task)
    if limit:
        td.splits = type(td.splits)({k: v.select(range(min(limit, len(v))))
                                     for k, v in td.splits.items()})
    if td.kind == "qa":
        raise ValueError("feature-based QA is done as a frozen linear probe in train.py")
    run_id = run_id or f"frozen_{estimator}_{pooling}_{model_name.split('/')[-1]}"

    with Stopwatch() as sw_extract:
        Xtr, ytr, ltr = extract(td, "train", model_name, pooling=pooling)
        Xte, yte, lte = extract(td, "test", model_name, pooling=pooling)

    rng = np.random.default_rng(seed)
    if td.kind == "token" and len(Xtr) > MAX_TRAIN_TOKENS:
        idx = rng.choice(len(Xtr), MAX_TRAIN_TOKENS, replace=False)
        Xfit, yfit = Xtr[idx], ytr[idx]
        subsample_note = f"fit on {MAX_TRAIN_TOKENS:,} of {len(Xtr):,} training tokens"
    else:
        Xfit, yfit, subsample_note = Xtr, ytr, ""

    clf = build_estimator(estimator, seed)
    with Stopwatch() as sw_fit:
        clf.fit(Xfit, yfit)
    with Stopwatch() as sw_pred:
        pred = clf.predict(Xte)

    if td.kind == "sequence":
        metrics = sequence_metrics(yte, pred, td.labels)
    else:
        true_seqs = _regroup(yte, lte, td.labels)
        pred_seqs = _regroup(pred, lte, td.labels)
        metrics = (ner_metrics(true_seqs, pred_seqs) if task == "ner"
                   else pos_metrics(true_seqs, pred_seqs, td.labels))

    n_params = int(np.prod(getattr(clf, "coef_", np.zeros((0, 0))).shape) +
                   getattr(clf, "intercept_", np.zeros(0)).size) or None
    rec = RunRecord(
        task=task, run_id=run_id, method="frozen", model_name=model_name,
        head=f"sklearn:{estimator}",
        hyperparams={"estimator": estimator, "pooling": pooling,
                     "params": clf.get_params(), "max_length": MAX_LEN[task]},
        params={"total_params": BODY_PARAMS.get(model_name), "trainable_params": 0,
                "trainable_pct": 0.0, "learner_params": n_params,
                "note": "0 BERT parameters updated; the learner sits on top of frozen features"},
        metrics=metrics,
        train_seconds=sw_fit.seconds, eval_seconds=sw_pred.seconds,
        epoch_times=[{"stage": "feature extraction", "seconds": sw_extract.seconds},
                     {"stage": "estimator fit", "seconds": sw_fit.seconds},
                     {"stage": "predict test", "seconds": sw_pred.seconds}],
        dataset=dataset_card(td), seed=seed,
        notes=" ".join(x for x in (notes, subsample_note) if x))
    rec.save()
    return rec
