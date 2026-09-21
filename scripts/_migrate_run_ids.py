#!/usr/bin/env python
"""One-off: rename token-task feature results written under the old `_cls_` run id.

Pooling only means something for sequence classification; token tasks always take the
first-subword vector, so a run id saying "cls" misdescribes what was computed.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
moved = 0
for task in ("ner", "pos"):
    for old in (ROOT / "results" / task).glob("frozen_sk-*_cls_*.json"):
        new = old.with_name(old.name.replace("_cls_", "_firstsub_"))
        rec = json.loads(old.read_text())
        rec["run_id"] = new.stem
        rec.setdefault("hyperparams", {}).pop("pooling", None)
        new.write_text(json.dumps(rec, indent=2, default=str))
        old.unlink()
        print(f"renamed {task}/{old.name} -> {new.name}")
        moved += 1
print(f"[migrate] {moved} file(s)")
