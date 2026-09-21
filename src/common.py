"""Shared utilities: seeding, device selection, freezing schemes, parameter groups, timing."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CACHE = ROOT / ".cache_features"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we touch. Called at the top of every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def precision_flags() -> dict[str, bool]:
    """Mixed precision that the device actually supports.

    MPS takes bfloat16 and roughly doubles throughput. On CUDA, bfloat16 needs Ampere or
    newer - a P100 or T4 does not have it, and asking for it makes TrainingArguments refuse
    to start - so those fall back to float16.
    """
    dev = get_device()
    if dev.type == "mps":
        return {"bf16": True, "fp16": False}
    if dev.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported()
        return {"bf16": bf16, "fp16": not bf16}
    return {"bf16": False, "fp16": False}


def autocast_dtype() -> torch.dtype | None:
    """The dtype to run frozen-feature extraction under, or None to stay in float32."""
    flags = precision_flags()
    return torch.bfloat16 if flags["bf16"] else (torch.float16 if flags["fp16"] else None)


def free_device_memory() -> None:
    """Drop cached device allocations between runs.

    A grid runs many experiments in one process; on MPS the allocator keeps freed blocks
    cached, and across twenty runs that grew until the machine swapped and training slowed
    by two orders of magnitude.
    """
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment_info() -> dict[str, Any]:
    dev = get_device()
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(dev),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if dev.type == "mps":
        try:
            info["chip"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - informational only
            pass
    elif dev.type == "cuda":
        info["chip"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------- #
# Adaptation ladder: which parameters are trainable
# --------------------------------------------------------------------------- #

def _encoder(model: torch.nn.Module):
    """The BERT/DistilBERT body inside a *ForXxx head model."""
    for name in ("bert", "distilbert", "roberta", "base_model"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("no transformer body found on model")


def _layers(body: torch.nn.Module):
    """The list of encoder layers, whatever the architecture calls it."""
    if hasattr(body, "encoder") and hasattr(body.encoder, "layer"):
        return body.encoder.layer          # BERT
    if hasattr(body, "transformer"):
        return body.transformer.layer      # DistilBERT
    raise AttributeError("no encoder layers found on body")


def apply_adaptation(model: torch.nn.Module, method: str,
                     pooler_as_head: bool = False) -> dict[str, Any]:
    """Freeze/unfreeze according to a rung of the ladder.

    ``frozen``          - body entirely frozen, only the head trains (feature-based).
    ``partial_ftN``     - head + top N encoder layers.
    ``full_ft``         - everything.

    ``pooler_as_head`` only matters for sequence classification. BertForSequenceClassification
    does not classify the raw [CLS] state: it classifies ``tanh(dense([CLS]))``, BERT's pooler,
    which was pretrained for next-sentence prediction. Freezing it makes a "linear probe" read
    a tanh-saturated NSP vector instead of the sentence representation - so this flag lets the
    pooler count as head (still zero encoder parameters updated). Both variants are measured.
    """
    body = _encoder(model)
    layers = _layers(body)
    n_layers = len(layers)

    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        top_n = n_layers
    else:
        for p in body.parameters():
            p.requires_grad = False
        if method.startswith("partial_ft"):
            top_n = int(method.replace("partial_ft", ""))
            for layer in layers[n_layers - top_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        elif method == "frozen":
            top_n = 0
            if pooler_as_head and getattr(body, "pooler", None) is not None:
                for p in body.pooler.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"unknown adaptation method: {method}")
        # The head always trains; everything outside the body is head.
        head_of_body = (body.pooler.parameters() if pooler_as_head and method == "frozen"
                        and getattr(body, "pooler", None) is not None else ())
        body_ids = {id(p) for p in body.parameters()} - {id(p) for p in head_of_body}
        for p in model.parameters():
            if id(p) not in body_ids:
                p.requires_grad = True

    if method == "frozen":
        # A feature extractor must be deterministic: pin the encoder in eval mode so its
        # dropout never fires, and keep Trainer's model.train() from switching it back.
        # The pooler, when it counts as head, keeps its normal train/eval behaviour.
        frozen_part = body.encoder if pooler_as_head and hasattr(body, "encoder") else body
        frozen_part.eval()
        frozen_part.train = lambda mode=True, _b=frozen_part: _b

    return {"method": method, "encoder_layers": n_layers, "unfrozen_top_layers": top_n,
            "pooler_as_head": bool(pooler_as_head)}


def param_groups(model: torch.nn.Module, head_lr: float, body_lr: float) -> list[dict]:
    """Two parameter groups: a fresh head takes big steps, pretrained weights take small ones.

    A trainable parameter inside the body goes in the body group *unless* it is the pooler
    being used as head, which is re-initialised in spirit and takes the head rate.
    """
    body = _encoder(model)
    body_ids = {id(p) for p in body.parameters()}
    pooler = getattr(body, "pooler", None)
    if pooler is not None and all(p.requires_grad for p in pooler.parameters()) \
            and not any(p.requires_grad for p in _layers(body).parameters()):
        body_ids -= {id(p) for p in pooler.parameters()}
    head, body = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (body if id(p) in body_ids else head).append(p)
    groups = []
    if head:
        groups.append({"params": head, "lr": head_lr, "name": "head"})
    if body:
        groups.append({"params": body, "lr": body_lr, "name": "body"})
    return groups


def count_params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable,
            "trainable_pct": round(100.0 * trainable / total, 3)}


def head_param_count(model: torch.nn.Module) -> int:
    body_ids = {id(p) for p in _encoder(model).parameters()}
    return sum(p.numel() for p in model.parameters() if id(p) not in body_ids)


# --------------------------------------------------------------------------- #
# Timing / logging
# --------------------------------------------------------------------------- #

class EpochTimer(TrainerCallback):
    """Records wall-clock seconds per epoch and per step, plus the log history."""

    def __init__(self) -> None:
        self.epochs: list[dict[str, float]] = []
        self._t0 = None
        self._step0 = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        self._step0 = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        dt = time.perf_counter() - self._t0
        steps = max(1, state.global_step - self._step0)
        self.epochs.append({
            "epoch": round(state.epoch, 3),
            "seconds": round(dt, 2),
            "steps": steps,
            "sec_per_step": round(dt / steps, 4),
        })


@dataclass
class RunRecord:
    """One row of the experiment grid; serialised to results/<task>/<run_id>.json."""
    task: str
    run_id: str
    method: str
    model_name: str
    head: str = ""
    hyperparams: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    train_log: list[dict[str, Any]] = field(default_factory=list)
    epoch_times: list[dict[str, Any]] = field(default_factory=list)
    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    dataset: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=environment_info)
    seed: int = SEED
    notes: str = ""

    def save(self) -> Path:
        out = RESULTS / self.task
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.run_id}.json"
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))
        return path


def load_results(task: str | None = None) -> list[dict[str, Any]]:
    pattern = f"{task}/*.json" if task else "*/*.json"
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


class Stopwatch:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self._t0, 2)
