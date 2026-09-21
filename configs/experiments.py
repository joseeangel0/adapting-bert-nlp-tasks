"""The experiment grid: every run in the assignment, with its hyper-parameters.

Two rules shape it: at least two adaptation methods per task, and all three rungs of the
ladder (feature-based, partial fine-tuning, full fine-tuning) appear somewhere. Learning
rates follow the two-group rule - 1e-3 for a freshly initialised head, 2e-5/3e-5 for
pretrained encoder weights.

`kind="feature"` runs go through src/feature_based.py (frozen BERT + scikit-learn).
`kind="train"`   runs go through src/train.py (frozen probe / partial / full fine-tuning).
"""

SEED = 42
BODY = "bert-base-uncased"

GRID = [
    # ---------------- AG News: topic classification, 4 classes ------------------- #
    dict(task="agnews", kind="feature", estimator="logreg", pooling="cls"),
    dict(task="agnews", kind="feature", estimator="logreg", pooling="mean"),
    dict(task="agnews", kind="feature", estimator="linsvm", pooling="cls"),
    dict(task="agnews", kind="feature", estimator="rf", pooling="cls"),
    dict(task="agnews", kind="train", method="frozen", head="linear",
         epochs=5, batch_size=32, head_lr=1e-3),
    dict(task="agnews", kind="train", method="frozen", head="mlp",
         epochs=5, batch_size=32, head_lr=1e-3),
    # Same probe, but BERT's NSP pooler counts as head instead of as frozen body: isolates
    # how much of the frozen-probe score is lost to the tanh pooler bottleneck.
    dict(task="agnews", kind="train", method="frozen", head="linear", pooler_as_head=True,
         epochs=5, batch_size=32, head_lr=1e-3),
    dict(task="agnews", kind="train", method="partial_ft2", head="linear",
         epochs=3, batch_size=32, head_lr=1e-3, body_lr=3e-5),
    dict(task="agnews", kind="train", method="full_ft", head="linear",
         epochs=2, batch_size=32, head_lr=1e-3, body_lr=2e-5),

    # ---------------- CoNLL-2003: named entity recognition ----------------------- #
    dict(task="ner", kind="feature", estimator="logreg", pooling="cls"),
    dict(task="ner", kind="train", method="frozen", head="linear",
         epochs=5, batch_size=32, head_lr=1e-3),
    dict(task="ner", kind="train", method="partial_ft4", head="linear",
         epochs=3, batch_size=32, head_lr=1e-3, body_lr=3e-5),
    dict(task="ner", kind="train", method="full_ft", head="linear",
         epochs=3, batch_size=32, head_lr=1e-3, body_lr=2e-5),

    # ---------------- UD English-EWT: part-of-speech tagging --------------------- #
    dict(task="pos", kind="feature", estimator="logreg", pooling="cls"),
    dict(task="pos", kind="train", method="frozen", head="linear",
         epochs=5, batch_size=32, head_lr=1e-3),
    dict(task="pos", kind="train", method="partial_ft2", head="linear",
         epochs=3, batch_size=32, head_lr=1e-3, body_lr=3e-5),
    dict(task="pos", kind="train", method="full_ft", head="linear",
         epochs=3, batch_size=32, head_lr=1e-3, body_lr=2e-5),

    # ---------------- SQuAD v1.1: extractive question answering ------------------ #
    dict(task="qa", kind="train", method="frozen", head="linear",
         epochs=3, batch_size=16, head_lr=1e-3),
    dict(task="qa", kind="train", method="partial_ft4", head="linear",
         epochs=2, batch_size=16, head_lr=1e-3, body_lr=3e-5),
    dict(task="qa", kind="train", method="full_ft", head="linear",
         epochs=2, batch_size=16, head_lr=1e-3, body_lr=3e-5),
]

# Optional size benchmark, only if there is compute left (Part 1, "optional benchmarks").
OPTIONAL = [
    dict(task="agnews", kind="train", method="full_ft", head="linear", model_name="distilbert-base-uncased",
         epochs=2, batch_size=32, head_lr=1e-3, body_lr=2e-5),
    dict(task="ner", kind="train", method="full_ft", head="linear", model_name="distilbert-base-uncased",
         epochs=3, batch_size=32, head_lr=1e-3, body_lr=2e-5),
]


def run_id(cfg: dict) -> str:
    body = cfg.get("model_name", BODY).split("/")[-1]
    if cfg["kind"] == "feature":
        return f"frozen_sk-{cfg['estimator']}_{cfg.get('pooling', 'cls')}_{body}"
    suffix = "-pooler" if cfg.get("pooler_as_head") else ""
    return f"{cfg['method']}_{cfg.get('head', 'linear')}{suffix}_{body}"
