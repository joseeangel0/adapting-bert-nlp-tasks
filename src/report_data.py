"""Aggregation layer: turns results/*/*.json into the rows the figures and the report use."""
from __future__ import annotations

from typing import Any

from .common import load_results

TASK_TITLES = {
    "agnews": "Topic classification (AG News)",
    "ner": "Named entity recognition (CoNLL-2003)",
    "pos": "Part-of-speech tagging (UD English-EWT)",
    "qa": "Extractive QA (SQuAD v1.1)",
}
TASK_ORDER = ["agnews", "ner", "pos", "qa"]

# The headline metric per task, and the name it gets in tables.
HEADLINE = {
    "agnews": ("accuracy", "Accuracy"),
    "ner": ("entity_f1", "Entity F1"),
    "pos": ("token_accuracy", "Token accuracy"),
    "qa": ("f1", "F1"),
}
SECONDARY = {
    "agnews": ("macro_f1", "Macro F1"),
    "ner": ("token_accuracy", "Token accuracy"),
    "pos": ("macro_f1", "Macro F1"),
    "qa": ("exact_match", "Exact match"),
}

# The three rungs of the ladder; colors come from the validated categorical palette.
RUNGS = {"frozen": "Feature-based", "partial": "Partial fine-tuning", "full": "Full fine-tuning"}
RUNG_COLOR = {"frozen": "#2a78d6", "partial": "#eb6834", "full": "#1baf7a"}


def rung(rec: dict) -> str:
    m = rec["method"]
    return "frozen" if m == "frozen" else ("full" if m == "full_ft" else "partial")


def method_label(rec: dict) -> str:
    m, head = rec["method"], rec.get("head", "")
    if m == "frozen":
        consumer = {"sklearn:logreg": "logistic regression", "sklearn:linsvm": "linear SVM",
                    "sklearn:rf": "random forest", "linear": "linear probe",
                    "mlp": "MLP probe"}.get(head, head)
        pooling = rec.get("hyperparams", {}).get("pooling")
        suffix = f", {pooling}-pooled" if pooling and pooling != "cls" else ""
        if rec.get("params", {}).get("pooler_as_head"):
            suffix += ", pooler as head"
        return f"Feature-based ({consumer}{suffix})"
    if m.startswith("partial_ft"):
        return f"Partial FT (top {m.replace('partial_ft', '')} layers)"
    return "Full fine-tuning"


def score(rec: dict, which: str = "headline") -> float | None:
    key = (HEADLINE if which == "headline" else SECONDARY)[rec["task"]][0]
    v = rec["metrics"].get(key)
    if v is None:
        return None
    return v * 100 if rec["task"] in {"agnews", "ner", "pos"} else v


def trainable(rec: dict) -> int:
    p = rec["params"]
    return p.get("trainable_params", 0) or (p.get("learner_params") or 0)


def train_minutes(rec: dict) -> float:
    extract = sum(e.get("seconds", 0) for e in rec.get("epoch_times", [])
                  if e.get("stage") == "feature extraction")
    return round((rec.get("train_seconds", 0) + extract) / 60, 2)


def rows(task: str | None = None) -> list[dict[str, Any]]:
    out = []
    for rec in load_results(task):
        out.append({
            "task": rec["task"], "run_id": rec["run_id"], "rung": rung(rec),
            "method": method_label(rec), "model": rec["model_name"],
            "headline": score(rec, "headline"), "secondary": score(rec, "secondary"),
            "headline_name": HEADLINE[rec["task"]][1],
            "secondary_name": SECONDARY[rec["task"]][1],
            "trainable": trainable(rec), "trainable_pct": rec["params"].get("trainable_pct", 0.0),
            "minutes": train_minutes(rec), "raw": rec,
        })
    order = {"frozen": 0, "partial": 1, "full": 2}
    out.sort(key=lambda r: (TASK_ORDER.index(r["task"]), order[r["rung"]], -(r["headline"] or 0)))
    return out


def best_per_task() -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows():
        if r["headline"] is None:
            continue
        if r["task"] not in best or r["headline"] > best[r["task"]]["headline"]:
            best[r["task"]] = r
    return best


def curve(rec: dict, key: str) -> list[tuple[float, float]]:
    """(epoch or step, value) pairs from the Trainer log history."""
    pts = []
    for entry in rec.get("train_log", []):
        if key in entry and entry.get("epoch") is not None:
            x = entry["step"] if key == "loss" else entry["epoch"]
            pts.append((x, entry[key]))
    return pts
