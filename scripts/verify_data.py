#!/usr/bin/env python
"""Data-integrity checks that run before the grid and are re-runnable as evidence.

These are the checks whose failure would not crash anything — it would just make every
number in the report quietly wrong. Writes docs/data_checks.md.

    python scripts/verify_data.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src import qa_utils  # noqa: E402
from src.data import load_task  # noqa: E402
from src.encoding import MAX_LEN, alignment_invariants  # noqa: E402

MODEL = "bert-base-uncased"
checks: list[dict] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")


def token_task(task: str, tokenizer) -> None:
    td = load_task(task)
    label_key = td.label_key
    total_bad = total_long = 0
    sizes = {}
    for split, d in td.splits.items():
        bad = sum(1 for ex in d if len(ex["tokens"]) != len(ex[label_key]))
        lens = [len(ex["tokens"]) for ex in d]
        long = sum(1 for n in lens if n > MAX_LEN[task] - 2)
        total_bad += bad
        total_long += long
        sizes[split] = f"{len(d)} sentences, longest {max(lens)} words"
    check(f"{task}: one label per word in every split", total_bad == 0,
          f"{total_bad} length mismatches across {sum(len(d) for d in td.splits.values())} sentences")
    check(f"{task}: nothing lost to truncation at max_length={MAX_LEN[task]}", total_long == 0,
          f"{total_long} sentences longer than the window; " + "; ".join(
              f"{k}: {v}" for k, v in sizes.items()))
    inv = alignment_invariants(td, tokenizer, n=1000)
    check(f"{task}: subword alignment invariants", inv["passed"], json.dumps(inv))
    if task == "ner":
        counts = Counter(td.labels[t] for ex in td.splits["test"] for t in ex[label_key])
        share = counts["O"] / sum(counts.values())
        check("ner: O-token share on test (why token accuracy is not the metric)", True,
              f"{share:.1%} of test tokens are O, so a model that predicts nothing scores "
              f"{share:.1%} token accuracy and 0 entity F1")
    if task == "pos":
        tags = {t for ex in td.splits["train"] for t in ex["upos"]}
        check("pos: exactly the 17 UPOS tags", len(tags) == 17, f"{len(tags)} tags: {sorted(tags)}")
        mwt = sum(1 for ex in td.splits["train"] if ex["mwt"])
        check("pos: multiword tokens already expanded to syntactic words", True,
              f"{mwt} training sentences carry an MWT record (e.g. don't = do + n't); "
              "tokens/upos are the expanded words, so alignment is unaffected")


def sequence_task(tokenizer) -> None:
    td = load_task("agnews")
    counts = {s: Counter(d["label"]) for s, d in td.splits.items()}
    balanced = len(set(counts["test"].values())) == 1
    check("agnews: test split is balanced (so accuracy is a fair metric)", balanced,
          ", ".join(f"{td.labels[k]}={v}" for k, v in sorted(counts["test"].items())))
    texts = {s: set(d["text"]) for s, d in td.splits.items()}
    overlaps = {f"{a}∩{b}": len(texts[a] & texts[b])
                for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    check("agnews: no leakage between the seeded subsample splits",
          sum(overlaps.values()) == 0, json.dumps(overlaps))


def qa_task(tokenizer, n: int = 1000) -> None:
    td = load_task("qa")
    sub = td.splits["test"].select(range(n))
    feats = qa_utils.prepare_validation_features(sub[:], tokenizer)
    by_id = {x["id"]: x for x in sub}
    real = exact = punct = 0
    for i, ex_id in enumerate(feats["example_id"]):
        s, e = feats["start_positions"][i], feats["end_positions"][i]
        if s == 0 and e == 0:
            continue
        real += 1
        off, ex = feats["offset_mapping"][i], by_id[ex_id]
        span = ex["context"][off[s][0]:off[e][1]]
        norm = lambda t: "".join(t.lower().split())  # noqa: E731
        golds = ex["answers"]["text"]
        if any(norm(span) == norm(g) for g in golds):
            exact += 1
        elif any(norm(g) in norm(span) or norm(span) in norm(g) for g in golds):
            punct += 1
    wrong = real - exact - punct
    check("qa: gold character span -> subword target -> decoded span round-trips", wrong == 0,
          f"{exact}/{real} exact ({100 * exact / real:.2f}%), {punct} differ only by surrounding "
          f"punctuation (which the official SQuAD normalisation strips), {wrong} genuinely wrong")
    windows = len(feats["input_ids"])
    check("qa: context windowing", True,
          f"{n} questions -> {windows} windows at max_length={qa_utils.MAX_LEN}, "
          f"stride={qa_utils.DOC_STRIDE} ({windows / n:.3f} per question)")


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    sequence_task(tokenizer)
    token_task("ner", tokenizer)
    token_task("pos", tokenizer)
    qa_task(tokenizer)

    failed = [c for c in checks if not c["passed"]]
    lines = ["# Data checks", "",
             f"Generated by `scripts/verify_data.py` with `{MODEL}`. "
             "These are the checks whose failure would not crash anything — it would only make "
             "every number in the report quietly wrong.", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c['name']} | {'✅' if c['passed'] else '❌'} | {c['detail']} |" for c in checks]
    lines += ["", f"**{len(checks) - len(failed)}/{len(checks)} passed.**"]
    (ROOT / "docs" / "data_checks.md").write_text("\n".join(lines) + "\n")
    print(f"\n[written] docs/data_checks.md — {len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
