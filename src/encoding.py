"""Tokenization and the subword/label alignment that every token-level task needs.

The convention (assignment + HF token-classification recipe): the word's label goes
on its *first* subword; continuation subwords, [CLS], [SEP] and padding get -100, the
index CrossEntropyLoss ignores. `alignment_report` prints tokens next to labels so the
alignment can be eyeballed before any training starts.
"""
from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from .data import TaskData

IGNORE = -100
MAX_LEN = {"agnews": 128, "ner": 192, "pos": 192, "qa": 384}


def tokenize_sequence(td: TaskData, tokenizer, max_length: int | None = None) -> DatasetDict:
    max_length = max_length or MAX_LEN[td.name]

    def fn(batch):
        enc = tokenizer(batch[td.text_key], truncation=True, max_length=max_length)
        enc["labels"] = batch[td.label_key]
        return enc

    return DatasetDict({
        k: v.map(fn, batched=True, remove_columns=v.column_names, desc=f"tokenize {k}")
        for k, v in td.splits.items()
    })


def align_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """One label per word -> one label per subword, -100 everywhere else."""
    out, previous = [], None
    for wid in word_ids:
        if wid is None:
            out.append(IGNORE)
        elif wid != previous:
            out.append(word_labels[wid])
        else:
            out.append(IGNORE)
        previous = wid
    return out


def tokenize_token_task(td: TaskData, tokenizer, max_length: int | None = None) -> DatasetDict:
    max_length = max_length or MAX_LEN[td.name]

    def fn(batch):
        enc = tokenizer(batch[td.text_key], is_split_into_words=True,
                        truncation=True, max_length=max_length)
        enc["labels"] = [align_labels(enc.word_ids(i), labels)
                         for i, labels in enumerate(batch[td.label_key])]
        return enc

    return DatasetDict({
        k: v.map(fn, batched=True, remove_columns=v.column_names, desc=f"tokenize {k}")
        for k, v in td.splits.items()
    })


def tokenize_task(td: TaskData, tokenizer, max_length: int | None = None) -> DatasetDict:
    if td.kind == "sequence":
        return tokenize_sequence(td, tokenizer, max_length)
    if td.kind == "token":
        return tokenize_token_task(td, tokenizer, max_length)
    raise ValueError(f"use qa_utils for kind={td.kind}")


def alignment_report(td: TaskData, tokenizer, n_examples: int = 2) -> str:
    """Human-readable check: subword tokens next to the label that landed on them."""
    lines = [f"=== {td.name}: subword/label alignment ({tokenizer.name_or_path}) ==="]
    split = td.splits["train"]
    for i in range(n_examples):
        ex = split[i]
        enc = tokenizer(ex[td.text_key], is_split_into_words=True, truncation=True,
                        max_length=MAX_LEN[td.name])
        labels = align_labels(enc.word_ids(0), ex[td.label_key])
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        lines.append(f"\n-- example {i} ({len(ex[td.text_key])} words -> {len(toks)} subwords) --")
        lines.append(f"{'subword':<16}{'word_id':>8}{'label':>8}  tag")
        for tok, wid, lab in zip(toks, enc.word_ids(0), labels):
            tag = "(ignored)" if lab == IGNORE else td.labels[lab]
            lines.append(f"{tok:<16}{str(wid):>8}{lab:>8}  {tag}")
        n_words, n_labelled = len(ex[td.text_key]), sum(l != IGNORE for l in labels)
        lines.append(f"words={n_words}  labelled positions={n_labelled}  "
                     f"{'OK' if n_words == n_labelled else 'TRUNCATED (expected for long sentences)'}")
    return "\n".join(lines)


def alignment_invariants(td: TaskData, tokenizer, n: int = 500) -> dict[str, Any]:
    """Machine check over n examples: len(labels)==len(input_ids), one label per word."""
    split, bad_len, bad_count, truncated = td.splits["train"], 0, 0, 0
    for i in range(min(n, len(split))):
        ex = split[i]
        enc = tokenizer(ex[td.text_key], is_split_into_words=True, truncation=True,
                        max_length=MAX_LEN[td.name])
        labels = align_labels(enc.word_ids(0), ex[td.label_key])
        if len(labels) != len(enc["input_ids"]):
            bad_len += 1
        n_labelled = sum(l != IGNORE for l in labels)
        if n_labelled != len(ex[td.text_key]):
            truncated += 1
            if len(enc["input_ids"]) < MAX_LEN[td.name]:
                bad_count += 1      # short sentence that still lost labels = a real bug
    return {"checked": min(n, len(split)), "length_mismatches": bad_len,
            "truncated_sentences": truncated, "unexplained_label_loss": bad_count,
            "passed": bad_len == 0 and bad_count == 0}
