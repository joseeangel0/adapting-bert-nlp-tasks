"""SQuAD v1.1 feature preparation and span post-processing.

Extractive QA is the token-level task where alignment is subtlest: the gold answer is a
character span in the context, and it has to be converted into *subword* start/end indices
in a window that may not even contain the answer (long contexts are split with a stride).
Windows without the answer are trained toward [CLS], the no-answer position.
Follows the reference recipe from the HuggingFace question-answering example.
"""
from __future__ import annotations

import collections
from typing import Any

import numpy as np
from datasets import Dataset

MAX_LEN = 384
DOC_STRIDE = 128
N_BEST = 20
MAX_ANSWER_LEN = 30


def _span_targets(enc, offsets_all, sample_map, examples, tokenizer) -> tuple[list, list]:
    """Character answer span -> subword (start, end) per window; [CLS] when the window misses it."""
    starts, ends = [], []
    for i, offsets in enumerate(offsets_all):
        input_ids = enc["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = enc.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]

        if len(answers["answer_start"]) == 0:
            starts.append(cls_index); ends.append(cls_index); continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        tok_start = 0
        while sequence_ids[tok_start] != 1:
            tok_start += 1
        tok_end = len(input_ids) - 1
        while sequence_ids[tok_end] != 1:
            tok_end -= 1

        if not (offsets[tok_start][0] <= start_char and offsets[tok_end][1] >= end_char):
            starts.append(cls_index); ends.append(cls_index)      # answer not in this window
        else:
            while tok_start < len(offsets) and offsets[tok_start][0] <= start_char:
                tok_start += 1
            starts.append(tok_start - 1)
            while offsets[tok_end][1] >= end_char:
                tok_end -= 1
            ends.append(tok_end + 1)
    return starts, ends


def _encode(examples, tokenizer, max_length: int, stride: int):
    questions = [q.lstrip() for q in examples["question"]]
    enc = tokenizer(questions, examples["context"], truncation="only_second",
                    max_length=max_length, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding=False)
    return enc, enc.pop("overflow_to_sample_mapping")


def prepare_train_features(examples, tokenizer, max_length: int = MAX_LEN,
                           stride: int = DOC_STRIDE) -> dict[str, Any]:
    enc, sample_map = _encode(examples, tokenizer, max_length, stride)
    offsets_all = enc.pop("offset_mapping")
    enc["start_positions"], enc["end_positions"] = _span_targets(
        enc, offsets_all, sample_map, examples, tokenizer)
    return enc


def prepare_validation_features(examples, tokenizer, max_length: int = MAX_LEN,
                                stride: int = DOC_STRIDE) -> dict[str, Any]:
    """Evaluation windows: gold positions *and* the offsets needed to decode a span back.

    The gold positions are kept so the evaluation pass also yields a loss and so
    Trainer calls compute_metrics - which is where the spans are decoded, in a single
    forward pass over the split rather than two.
    """
    enc, sample_map = _encode(examples, tokenizer, max_length, stride)
    offsets_all = enc["offset_mapping"]
    enc["start_positions"], enc["end_positions"] = _span_targets(
        enc, offsets_all, sample_map, examples, tokenizer)
    enc["example_id"] = [examples["id"][sample_map[i]] for i in range(len(enc["input_ids"]))]
    # Keep only context offsets; a span must never be decoded out of the question.
    enc["offset_mapping"] = [
        [off if enc.sequence_ids(i)[k] == 1 else None for k, off in enumerate(offsets)]
        for i, offsets in enumerate(offsets_all)
    ]
    return enc


def postprocess_qa(examples: Dataset, features: Dataset, raw_predictions,
                   n_best: int = N_BEST, max_answer_len: int = MAX_ANSWER_LEN) -> dict[str, str]:
    """Turn start/end logits into one answer string per question."""
    start_logits, end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat_example_id in enumerate(features["example_id"]):
        features_per_example[example_id_to_index[feat_example_id]].append(i)

    predictions = {}
    offset_column = features["offset_mapping"]
    for example_index, example in enumerate(examples):
        context = example["context"]
        best_text, best_score = "", -np.inf
        for feature_index in features_per_example[example_index]:
            sl, el = start_logits[feature_index], end_logits[feature_index]
            offsets = offset_column[feature_index]
            start_idx = np.argsort(sl)[-1: -n_best - 1: -1]
            end_idx = np.argsort(el)[-1: -n_best - 1: -1]
            for s in start_idx:
                for e in end_idx:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    score = sl[s] + el[e]
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0]: offsets[e][1]]
        predictions[example["id"]] = best_text
    return predictions
