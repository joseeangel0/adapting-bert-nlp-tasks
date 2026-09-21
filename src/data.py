"""Dataset loading for the four tasks, with fixed-seed subsampling.

Every split returned here is deterministic: the subsample indices come from a
seeded shuffle, so a rerun (or the teacher's replica) sees exactly the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import DatasetDict, load_dataset

from .common import SEED

# AG News has no dev split, so we carve one out of train to watch the epochs.
AGNEWS_TRAIN_N = 20_000
AGNEWS_DEV_N = 5_000
SQUAD_TRAIN_N = 15_000      # the assignment asks for ~15k
SQUAD_DEV_MONITOR_N = 3_000  # cheap per-epoch monitoring; final eval uses all 10 570

# lhoestq/conll2003 ships ner_tags as plain ints, so the canonical CoNLL-2003 tag order
# is pinned here and verified against known sentences in verify_conll_labels().
CONLL_NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                  "B-MISC", "I-MISC"]

UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
             "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


@dataclass
class TaskData:
    name: str
    kind: str                  # "sequence" | "token" | "qa"
    splits: DatasetDict        # keys: train / dev / test
    labels: list[str]
    text_key: str = ""
    label_key: str = ""
    source: str = ""
    license: str = ""
    citation: str = ""

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.splits.items()}


def load_agnews() -> TaskData:
    raw = load_dataset("fancyzhx/ag_news")
    pool = raw["train"].shuffle(seed=SEED)
    train = pool.select(range(AGNEWS_TRAIN_N))
    dev = pool.select(range(AGNEWS_TRAIN_N, AGNEWS_TRAIN_N + AGNEWS_DEV_N))
    return TaskData(
        name="agnews", kind="sequence",
        splits=DatasetDict(train=train, dev=dev, test=raw["test"]),
        labels=raw["train"].features["label"].names,
        text_key="text", label_key="label",
        source="fancyzhx/ag_news",
        license="Custom (AG News corpus, academic use)",
        citation="Zhang, Zhao & LeCun (2015), Character-level Convolutional Networks for Text Classification",
    )


def verify_conll_labels(split) -> None:
    """Fail loudly if the int->tag mapping ever stops matching the corpus.

    Sentence 0 is the shared task's own example: "EU rejects German call to boycott
    British lamb ." with EU=B-ORG and German/British=B-MISC.
    """
    ex = split[0]
    got = {tok: CONLL_NER_TAGS[tag] for tok, tag in zip(ex["tokens"], ex["ner_tags"])}
    expected = {"EU": "B-ORG", "German": "B-MISC", "British": "B-MISC", "rejects": "O"}
    bad = {k: (got.get(k), v) for k, v in expected.items() if got.get(k) != v}
    if bad:
        raise RuntimeError(f"CoNLL tag mapping mismatch (token: got, expected): {bad}")


def load_conll2003() -> TaskData:
    raw = load_dataset("lhoestq/conll2003")
    verify_conll_labels(raw["train"])
    return TaskData(
        name="ner", kind="token",
        splits=DatasetDict(train=raw["train"], dev=raw["validation"], test=raw["test"]),
        labels=CONLL_NER_TAGS,
        text_key="tokens", label_key="ner_tags",
        source="lhoestq/conll2003",
        license="Reuters corpus terms / research use (CoNLL-2003 shared task)",
        citation="Tjong Kim Sang & De Meulder (2003), CoNLL-2003 shared task: language-independent NER",
    )


def load_ud_ewt() -> TaskData:
    raw = load_dataset("universal-dependencies/universal_dependencies", "en_ewt")
    tag2id = {t: i for i, t in enumerate(UPOS_TAGS)}

    def encode(batch):
        return {"labels_str": batch["upos"],
                "pos_ids": [[tag2id[t] for t in seq] for seq in batch["upos"]]}

    splits = DatasetDict(
        train=raw["train"].map(encode, batched=True),
        dev=raw["dev"].map(encode, batched=True),
        test=raw["test"].map(encode, batched=True),
    )
    return TaskData(
        name="pos", kind="token", splits=splits, labels=UPOS_TAGS,
        text_key="tokens", label_key="pos_ids",
        source="universal-dependencies/universal_dependencies (config en_ewt)",
        license="CC BY-SA 4.0",
        citation="Silveira et al. (2014) / Universal Dependencies 2.x, UD_English-EWT",
    )


def load_squad() -> TaskData:
    raw = load_dataset("rajpurkar/squad")
    train = raw["train"].shuffle(seed=SEED).select(range(SQUAD_TRAIN_N))
    dev_monitor = raw["validation"].shuffle(seed=SEED).select(range(SQUAD_DEV_MONITOR_N))
    return TaskData(
        name="qa", kind="qa",
        splits=DatasetDict(train=train, dev=dev_monitor, test=raw["validation"]),
        labels=[], text_key="question", label_key="answers",
        source="rajpurkar/squad (v1.1)",
        license="CC BY-SA 4.0",
        citation="Rajpurkar et al. (2016), SQuAD: 100,000+ Questions for Machine Comprehension of Text",
    )


LOADERS = {"agnews": load_agnews, "ner": load_conll2003, "pos": load_ud_ewt, "qa": load_squad}


def load_task(task: str) -> TaskData:
    if task not in LOADERS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(LOADERS)}")
    return LOADERS[task]()


def dataset_card(td: TaskData) -> dict[str, Any]:
    return {"source": td.source, "license": td.license, "citation": td.citation,
            "sizes": td.sizes(), "num_labels": td.num_labels, "labels": td.labels}
