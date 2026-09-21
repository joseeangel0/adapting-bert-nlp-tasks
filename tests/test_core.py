"""Tests for the pieces whose failure would be invisible in the reported numbers.

A misaligned label, a mis-scored span or a freeze that silently leaves the body trainable
all produce a perfectly healthy-looking loss curve and a wrong result. These pin the
behaviour down instead.

    .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.common import apply_adaptation, count_params, param_groups
from src.encoding import IGNORE, align_labels
from src.metrics import (decode_token_predictions, ner_metrics, pos_metrics,
                         sequence_metrics, squad_metrics)


# --------------------------- subword / label alignment ---------------------- #

def test_label_goes_on_first_subword_only():
    # "playing" -> play + ##ing : one word, two subwords, one label.
    word_ids = [None, 0, 1, 1, 2, None]
    assert align_labels(word_ids, [5, 7, 3]) == [IGNORE, 5, 7, IGNORE, 3, IGNORE]


def test_specials_and_padding_are_ignored():
    labels = align_labels([None, 0, None, None], [4])
    assert labels == [IGNORE, 4, IGNORE, IGNORE]
    assert sum(l != IGNORE for l in labels) == 1


def test_alignment_length_always_matches_input():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_words = int(rng.integers(1, 12))
        word_ids = [None]
        for w in range(n_words):
            word_ids += [w] * int(rng.integers(1, 4))
        word_ids.append(None)
        out = align_labels(word_ids, list(range(n_words)))
        assert len(out) == len(word_ids)
        assert sum(l != IGNORE for l in out) == n_words


def test_ignored_positions_are_dropped_before_scoring():
    logits = np.zeros((1, 4, 3)); logits[0, :, 1] = 1.0     # predicts class 1 everywhere
    labels = np.array([[IGNORE, 2, IGNORE, 0]])
    true_seqs, pred_seqs = decode_token_predictions(logits, labels, ["A", "B", "C"])
    assert true_seqs == [["C", "A"]] and pred_seqs == [["B", "B"]]


# ------------------------------- metrics ------------------------------------ #

def test_ner_scores_a_half_recovered_entity_as_wrong():
    gold = [["B-PER", "I-PER", "O"]]
    half = [["B-PER", "O", "O"]]
    m = ner_metrics(gold, half)
    assert m["entity_f1"] == 0.0            # the span is wrong, not half right
    assert m["token_accuracy"] > 0.6        # ...while token accuracy calls it mostly fine


def test_ner_exact_span_is_right():
    gold = [["B-ORG", "I-ORG", "O"]]
    assert ner_metrics(gold, gold)["entity_f1"] == 1.0


def test_macro_f1_exposes_a_collapsed_class():
    labels = ["A", "B", "C", "D"]
    y_true = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    y_pred = [0] * 10 + [1] * 10 + [2] * 10 + [0] * 10      # class D never predicted
    m = sequence_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 0.75
    assert m["macro_f1"] < m["accuracy"]
    assert m["per_class"]["D"]["recall"] == 0.0


def test_pos_macro_f1_is_not_dominated_by_frequent_tags():
    labels = ["NOUN", "VERB", "X"]
    true_seqs = [["NOUN"] * 50 + ["VERB"] * 45 + ["X"] * 5]
    pred_seqs = [["NOUN"] * 50 + ["VERB"] * 45 + ["NOUN"] * 5]   # rare tag always missed
    m = pos_metrics(true_seqs, pred_seqs, labels)
    assert m["token_accuracy"] == 0.95
    assert m["per_tag"]["X"]["f1-score"] == 0.0
    assert m["macro_f1"] < 0.7


@pytest.mark.parametrize("pred,gold,em,f1_positive", [
    ("Denver Broncos", "Denver Broncos", 1.0, True),
    ("the Denver Broncos", "Denver Broncos", 1.0, True),      # article stripped
    ("Denver Broncos.", "Denver Broncos", 1.0, True),         # punctuation stripped
    ("denver broncos", "Denver Broncos", 1.0, True),          # lowercased
    ("Denver", "Denver Broncos", 0.0, True),                  # partial -> F1 only
    ("Carolina Panthers", "Denver Broncos", 0.0, False),
    ("", "Denver Broncos", 0.0, False),
])
def test_squad_official_normalisation(pred, gold, em, f1_positive):
    refs = [{"id": "q", "answers": {"text": [gold]}}]
    m = squad_metrics({"q": pred}, refs)
    assert m["exact_match"] == em * 100
    assert (m["f1"] > 0) is f1_positive


def test_squad_takes_the_best_of_several_gold_answers():
    refs = [{"id": "q", "answers": {"text": ["1929", "in 1929", "the year 1929"]}}]
    assert squad_metrics({"q": "in 1929"}, refs)["exact_match"] == 100.0


# ------------------------- the adaptation ladder ----------------------------- #

@pytest.fixture(scope="module")
def tiny_model():
    from transformers import BertConfig, BertForTokenClassification
    cfg = BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=4,
                     num_attention_heads=2, intermediate_size=32, num_labels=3)
    return lambda: BertForTokenClassification(cfg)


def test_frozen_trains_no_encoder_parameter(tiny_model):
    m = tiny_model()
    apply_adaptation(m, "frozen")
    assert not any(p.requires_grad for p in m.bert.parameters())
    assert all(p.requires_grad for p in m.classifier.parameters())
    assert count_params(m)["trainable_params"] == sum(p.numel() for p in m.classifier.parameters())


def test_frozen_body_stays_in_eval_so_features_are_deterministic(tiny_model):
    m = tiny_model()
    apply_adaptation(m, "frozen")
    m.train()                                   # what Trainer does every step
    assert not m.bert.training                  # ...must not wake the frozen body's dropout


def test_partial_unfreezes_exactly_the_top_layers(tiny_model):
    m = tiny_model()
    apply_adaptation(m, "partial_ft2")
    live = [i for i, l in enumerate(m.bert.encoder.layer)
            if any(p.requires_grad for p in l.parameters())]
    assert live == [2, 3]                       # top 2 of 4
    assert not any(p.requires_grad for p in m.bert.embeddings.parameters())


def test_full_trains_everything(tiny_model):
    m = tiny_model()
    apply_adaptation(m, "full_ft")
    assert all(p.requires_grad for p in m.parameters())


def test_two_parameter_groups_get_different_learning_rates(tiny_model):
    m = tiny_model()
    apply_adaptation(m, "partial_ft2")
    groups = {g["name"]: g["lr"] for g in param_groups(m, head_lr=1e-3, body_lr=2e-5)}
    assert groups == {"head": 1e-3, "body": 2e-5}


def test_frozen_run_has_only_a_head_group(tiny_model):
    m = tiny_model()
    apply_adaptation(m, "frozen")
    assert [g["name"] for g in param_groups(m, 1e-3, 2e-5)] == ["head"]


def test_unknown_method_is_rejected(tiny_model):
    with pytest.raises(ValueError):
        apply_adaptation(tiny_model(), "partial")


# ------------------------------ QA span decoding ----------------------------- #

def test_postprocess_picks_the_highest_scoring_valid_span():
    from datasets import Dataset
    from src.qa_utils import postprocess_qa
    context = "The Broncos won Super Bowl 50."
    examples = Dataset.from_dict({"id": ["q"], "context": [context],
                                  "question": ["Who won?"],
                                  "answers": [{"text": ["The Broncos"], "answer_start": [0]}]})
    offsets = [None, (0, 3), (4, 11), (12, 15), None]      # question/specials masked out
    features = Dataset.from_dict({"example_id": ["q"], "offset_mapping": [offsets]})
    start = np.full((1, 5), -9.0); end = np.full((1, 5), -9.0)
    start[0, 1] = 5.0      # "The"
    end[0, 2] = 5.0        # "Broncos"
    preds = postprocess_qa(examples, features, (start, end))
    assert preds["q"] == "The Broncos"


def test_postprocess_never_returns_a_reversed_or_masked_span():
    from datasets import Dataset
    from src.qa_utils import postprocess_qa
    examples = Dataset.from_dict({"id": ["q"], "context": ["alpha beta"],
                                  "question": ["?"],
                                  "answers": [{"text": ["beta"], "answer_start": [6]}]})
    offsets = [None, (0, 5), (6, 10)]
    features = Dataset.from_dict({"example_id": ["q"], "offset_mapping": [offsets]})
    start = np.array([[9.0, 1.0, 0.5]])       # highest start is the masked [CLS]
    end = np.array([[9.0, 0.1, 1.0]])
    preds = postprocess_qa(examples, features, (start, end))
    assert preds["q"] in {"alpha beta", "beta", "alpha"}   # never "" and never reversed
    assert preds["q"] != ""
