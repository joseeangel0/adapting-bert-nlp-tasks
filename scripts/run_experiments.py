#!/usr/bin/env python
"""Run the experiment grid. Resumable: a run whose results/<task>/<run_id>.json exists is skipped.

    python scripts/run_experiments.py                 # everything in configs/experiments.py
    python scripts/run_experiments.py --task qa       # one task
    python scripts/run_experiments.py --optional      # the DistilBERT size benchmark
    python scripts/run_experiments.py --force         # recompute even if results exist
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import feature_based, train           # noqa: E402
from src.common import RESULTS                 # noqa: E402

spec = importlib.util.spec_from_file_location("experiments", ROOT / "configs" / "experiments.py")
EXP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EXP)


def execute(cfg: dict, force: bool = False, limit: int | None = None) -> str:
    rid = EXP.run_id(cfg)
    task = cfg["task"]
    target = RESULTS / task / f"{rid}.json"
    if target.exists() and not force:
        return f"skip   {task:7s} {rid}"

    kwargs = {k: v for k, v in cfg.items() if k not in {"task", "kind"}}
    kwargs.setdefault("model_name", EXP.BODY)
    kwargs["seed"] = EXP.SEED
    kwargs["run_id"] = rid
    if limit:
        kwargs["limit"] = limit

    t0 = time.perf_counter()
    if cfg["kind"] == "feature":
        rec = feature_based.run(task, **kwargs)
    else:
        rec = train.run(task, kwargs.pop("method"), save_model=True, **kwargs)
    dt = time.perf_counter() - t0
    head = {k: v for k, v in rec.metrics.items() if isinstance(v, (int, float))}
    return f"done   {task:7s} {rid}  [{dt/60:.1f} min]  {head}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="*", help="restrict to these tasks")
    ap.add_argument("--optional", action="store_true", help="run the DistilBERT benchmark grid")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, help="rows per split (smoke test)")
    args = ap.parse_args()

    grid = EXP.OPTIONAL if args.optional else EXP.GRID
    if args.task:
        grid = [c for c in grid if c["task"] in args.task]

    print(f"[grid] {len(grid)} runs", flush=True)
    failures = 0
    for i, cfg in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] {cfg}", flush=True)
        try:
            print(execute(cfg, force=args.force, limit=args.limit), flush=True)
        except Exception:
            failures += 1
            traceback.print_exc()
            print(f"FAILED {cfg}", flush=True)
    print(f"\n[grid] finished with {failures} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
