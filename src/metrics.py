"""Per-task metrics, each chosen for what the task actually penalises.

sequence  : accuracy (AG News is balanced) + macro-F1 (catches a collapsed class)
            + per-class report + confusion matrix.
NER       : entity-level micro P/R/F1 via seqeval - a half-recovered entity is wrong,
            which token accuracy would happily call 93% correct. Token accuracy is kept
            as a secondary, and per-entity-type F1 exposes MISC/ORG weakness.
POS       : token accuracy (tags are dense and every token is a decision) + macro-F1
            over the 17 tags, because accuracy alone hides the rare tags (INTJ, SYM, X).
QA        : SQuAD exact match + token-level F1 with the official normalisation
            (lowercase, strip articles and punctuation), max over the gold answers.
"""
from __future__ import annotations

import collections
import re
import string
from typing import Any

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from .encoding import IGNORE


def sequence_metrics(y_true, y_pred, labels: list[str]) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rep = classification_report(y_true, y_pred, target_names=labels,
                                output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
        "per_class": {k: {m: round(float(v), 4) for m, v in d.items()}
                      for k, d in rep.items() if k in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_examples": int(len(y_true)),
    }


def decode_token_predictions(predictions, labels, label_names: list[str]):
    """Drop every -100 position, map ids to tag strings, keep the sentence grouping."""
    preds = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    true_seqs, pred_seqs = [], []
    for p_row, l_row in zip(preds, labels):
        t, q = [], []
        for p, l in zip(p_row, l_row):
            if l != IGNORE:
                t.append(label_names[l]); q.append(label_names[p])
        true_seqs.append(t); pred_seqs.append(q)
    return true_seqs, pred_seqs


def ner_metrics(true_seqs, pred_seqs) -> dict[str, Any]:
    from seqeval.metrics import (accuracy_score as seq_acc, classification_report as seq_rep,
                                 f1_score as seq_f1, precision_score as seq_p,
                                 recall_score as seq_r)
    rep = seq_rep(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    return {
        "entity_f1": round(float(seq_f1(true_seqs, pred_seqs, zero_division=0)), 5),
        "entity_precision": round(float(seq_p(true_seqs, pred_seqs, zero_division=0)), 5),
        "entity_recall": round(float(seq_r(true_seqs, pred_seqs, zero_division=0)), 5),
        "token_accuracy": round(float(seq_acc(true_seqs, pred_seqs)), 5),
        "per_entity": {k: {m: round(float(v), 4) for m, v in d.items()}
                       for k, d in rep.items() if not k.endswith("avg")},
        "n_sentences": len(true_seqs),
        "n_tokens": int(sum(len(s) for s in true_seqs)),
    }


def pos_metrics(true_seqs, pred_seqs, labels: list[str]) -> dict[str, Any]:
    y_true = [t for s in true_seqs for t in s]
    y_pred = [t for s in pred_seqs for t in s]
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    return {
        "token_accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                         zero_division=0)), 5),
        "per_tag": {k: {m: round(float(v), 4) for m, v in d.items()}
                    for k, d in rep.items() if k in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n_sentences": len(true_seqs),
        "n_tokens": len(y_true),
    }


# ---------------------------- SQuAD official metric ------------------------- #

def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_toks, g_toks = _normalize(pred).split(), _normalize(gold).split()
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p_toks), same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def squad_metrics(predictions: dict[str, str], references) -> dict[str, Any]:
    em = f1 = 0.0
    for ref in references:
        pred = predictions.get(ref["id"], "")
        golds = ref["answers"]["text"]
        em += max(float(_normalize(pred) == _normalize(g)) for g in golds)
        f1 += max(_f1(pred, g) for g in golds)
    n = len(references)
    return {"exact_match": round(100.0 * em / n, 3), "f1": round(100.0 * f1 / n, 3),
            "n_questions": n}
