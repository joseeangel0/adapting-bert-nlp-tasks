#!/usr/bin/env python
"""Print one batch with tokens next to their labels, before any training starts.

Writes docs/sanity_check.txt. Run it first: a silent misalignment trains against garbage
and still produces a loss curve that looks fine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from src.common import ROOT
from src.data import load_task
from src.encoding import alignment_invariants, alignment_report
from src.qa_utils import prepare_train_features

MODEL = "bert-base-uncased"


def qa_report(tokenizer, n: int = 2) -> str:
    td = load_task("qa")
    ex = td.splits["train"][:n]
    feats = prepare_train_features(ex, tokenizer)
    lines = ["=== qa: answer span -> subword start/end ==="]
    ok = 0
    for i in range(len(feats["input_ids"])):
        toks = tokenizer.convert_ids_to_tokens(feats["input_ids"][i])
        s, e = feats["start_positions"][i], feats["end_positions"][i]
        span = tokenizer.convert_tokens_to_string(toks[s:e + 1])
        sample = ex["answers"][min(i, n - 1)]["text"][0]
        lines += [f"\n-- feature {i} (window of {len(toks)} subwords) --",
                  f"question : {tokenizer.convert_tokens_to_string(toks[1:toks.index('[SEP]')])}",
                  f"start/end: {s}/{e}  ->  subwords {toks[s:e+1]}",
                  f"decoded  : {span!r}", f"gold     : {sample!r}"]
        norm = lambda t: "".join(t.lower().split())
        ok += int(norm(span) == norm(sample))
    lines.append(f"\nspans recovered: {ok}/{len(feats['input_ids'])}")
    return "\n".join(lines)


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    blocks, checks = [], {}
    for task in ("ner", "pos"):
        td = load_task(task)
        blocks.append(alignment_report(td, tokenizer))
        checks[task] = alignment_invariants(td, tokenizer)
        blocks.append(f"\ninvariant check: {json.dumps(checks[task])}\n")
    blocks.append(qa_report(tokenizer))

    text = "\n".join(blocks)
    out = ROOT / "docs" / "sanity_check.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(text[:4000])
    print(f"\n[written] {out}")
    failed = [t for t, c in checks.items() if not c["passed"]]
    if failed:
        print(f"ALIGNMENT FAILED for {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
