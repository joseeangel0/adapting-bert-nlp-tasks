"""Gradient-based runs: frozen head (linear probe), partial fine-tuning, full fine-tuning.

One code path for all four tasks - only the head class, the collator and the metric change.
Every run that touches pretrained weights uses two parameter groups: a big learning rate for
the freshly initialised head and a small one for the encoder, because one shared rate either
crawls for the head or destroys what pretraining built.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from .common import (EpochTimer, MODELS, RunRecord, Stopwatch, apply_adaptation,
                     count_params, get_device, head_param_count, param_groups, set_seed)
from .data import TaskData, dataset_card, load_task
from .encoding import MAX_LEN, tokenize_task
from .metrics import (decode_token_predictions, ner_metrics, pos_metrics,
                      sequence_metrics, squad_metrics)
from . import qa_utils


class MLPHeadWrapper(torch.nn.Module):
    """Replaces the linear classifier with a 768 -> hidden -> C MLP.

    Used only by feature-based runs, to test whether a non-linear consumer of the frozen
    representation buys anything over a linear one.
    """

    def __init__(self, in_features: int, num_labels: int, hidden: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, num_labels))

    def forward(self, x):
        return self.net(x)


class TwoGroupTrainer(Trainer):
    """Trainer with head/body parameter groups and no weight decay on bias & LayerNorm."""
    head_lr: float = 1e-3
    body_lr: float = 2e-5

    def create_optimizer(self):
        if self.optimizer is None:
            decay_blocked = ("bias", "LayerNorm.weight", "layer_norm.weight")
            named = dict(self.model.named_parameters())
            groups = []
            for g in param_groups(self.model, self.head_lr, self.body_lr):
                ids = {id(p) for p in g["params"]}
                decay = [p for n, p in named.items()
                         if id(p) in ids and not any(b in n for b in decay_blocked)]
                nodecay = [p for n, p in named.items()
                           if id(p) in ids and any(b in n for b in decay_blocked)]
                if decay:
                    groups.append({"params": decay, "lr": g["lr"],
                                   "weight_decay": self.args.weight_decay})
                if nodecay:
                    groups.append({"params": nodecay, "lr": g["lr"], "weight_decay": 0.0})
            cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            kwargs.pop("lr", None)
            kwargs.pop("weight_decay", None)
            self.optimizer = cls(groups, **kwargs)
        return self.optimizer


class QATrainer(TwoGroupTrainer):
    """Span post-processing happens inside compute_metrics, so each evaluation is one pass.

    The validation features carry no start/end positions - there is no loss to compute on
    them - so a separate scoring pass would be pure waste. `pack` is swapped between the
    development and test sets before each evaluation; predictions come back in feature
    order, which is the order postprocess_qa expects.
    """
    pack: tuple | None = None          # (examples, features_with_offsets)
    last_predictions: dict[str, str] | None = None

    def score(self, eval_pred):
        examples, feats = self.pack
        self.last_predictions = qa_utils.postprocess_qa(examples, feats, eval_pred.predictions)
        m = squad_metrics(self.last_predictions, examples)
        return {"exact_match": m["exact_match"], "f1": m["f1"]}


# --------------------------------------------------------------------------- #

def _attn_kwargs() -> dict[str, str]:
    """MPS has no fused attention kernel with dropout, so ask for the eager one there."""
    return {"attn_implementation": "eager"} if get_device().type == "mps" else {}


HEAD_CLASSES = {"sequence": AutoModelForSequenceClassification,
                "token": AutoModelForTokenClassification,
                "qa": AutoModelForQuestionAnswering}
MLP_HIDDEN = 512


def load_run_model(kind: str, path: str):
    """Reload a checkpoint this project saved, MLP probes included.

    ``from_pretrained`` rebuilds ``classifier`` as a plain Linear, which silently discards an
    MLP head's weights - a reloaded model would carry a randomly initialised head and score
    like chance. The head type is recorded in the config at build time, so it can be rebuilt
    here before its weights are restored.
    """
    cfg = AutoConfig.from_pretrained(path)
    model = HEAD_CLASSES[kind].from_pretrained(path, **_attn_kwargs())
    if getattr(cfg, "u2t01_head", "linear") == "mlp":
        from safetensors.torch import load_file
        model.classifier = MLPHeadWrapper(cfg.hidden_size, cfg.num_labels,
                                          hidden=getattr(cfg, "u2t01_head_hidden", MLP_HIDDEN))
        state = load_file(str(Path(path) / "model.safetensors"))
        model.classifier.load_state_dict(
            {k[len("classifier."):]: v for k, v in state.items() if k.startswith("classifier.")})
    return model


def build_model(td: TaskData, model_name: str, head: str = "linear"):
    attn = _attn_kwargs()
    if td.kind == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=td.num_labels,
            id2label={i: l for i, l in enumerate(td.labels)},
            label2id={l: i for i, l in enumerate(td.labels)}, **attn)
        if head == "mlp":
            model.classifier = MLPHeadWrapper(model.config.hidden_size, td.num_labels, MLP_HIDDEN)
    elif td.kind == "token":
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=td.num_labels,
            id2label={i: l for i, l in enumerate(td.labels)},
            label2id={l: i for i, l in enumerate(td.labels)}, **attn)
        if head == "mlp":
            model.classifier = MLPHeadWrapper(model.config.hidden_size, td.num_labels, MLP_HIDDEN)
    elif td.kind == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, **attn)
    else:
        raise ValueError(td.kind)
    # Recorded so load_run_model can rebuild a non-standard head before restoring its weights.
    model.config.u2t01_head = head
    if head == "mlp":
        model.config.u2t01_head_hidden = MLP_HIDDEN
    return model


def _qa_datasets(td: TaskData, tokenizer):
    train = td.splits["train"].map(
        lambda ex: qa_utils.prepare_train_features(ex, tokenizer), batched=True,
        remove_columns=td.splits["train"].column_names, desc="qa features train")
    packs = {}
    for split in ("dev", "test"):
        feats = td.splits[split].map(
            lambda ex: qa_utils.prepare_validation_features(ex, tokenizer), batched=True,
            remove_columns=td.splits[split].column_names, desc=f"qa features {split}")
        packs[split] = (td.splits[split], feats,
                        feats.remove_columns(["example_id", "offset_mapping"]))
    return train, packs


def run(task: str, method: str, model_name: str = "bert-base-uncased", *,
        head: str = "linear", epochs: float = 3, batch_size: int = 32,
        head_lr: float = 1e-3, body_lr: float = 2e-5, weight_decay: float = 0.01,
        warmup_ratio: float = 0.1, seed: int = 42, run_id: str | None = None,
        save_model: bool = False, eval_batch_size: int | None = None,
        limit: int | None = None, pooler_as_head: bool = False,
        notes: str = "") -> RunRecord:
    set_seed(seed)
    device = get_device()
    td = load_task(task)
    if limit:                      # smoke-test mode: a few hundred rows per split
        td.splits = type(td.splits)({k: v.select(range(min(limit, len(v))))
                                     for k, v in td.splits.items()})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    run_id = run_id or f"{method}_{head}_{model_name.split('/')[-1]}"
    eval_batch_size = eval_batch_size or batch_size * 2

    model = build_model(td, model_name, head)
    ladder = apply_adaptation(model, method, pooler_as_head=pooler_as_head)
    params = count_params(model) | ladder | {"head_params": head_param_count(model)}

    if td.kind == "qa":
        train_ds, packs = _qa_datasets(td, tokenizer)
        collator = DataCollatorWithPadding(tokenizer)
        eval_ds = packs["dev"][2]
        compute_metrics = None
    else:
        tok = tokenize_task(td, tokenizer)
        train_ds, eval_ds = tok["train"], tok["dev"]
        collator = (DataCollatorWithPadding(tokenizer) if td.kind == "sequence"
                    else DataCollatorForTokenClassification(tokenizer))

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            if td.kind == "sequence":
                m = sequence_metrics(labels, np.argmax(logits, axis=-1), td.labels)
                return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
            true_seqs, pred_seqs = decode_token_predictions(logits, labels, td.labels)
            if task == "ner":
                m = ner_metrics(true_seqs, pred_seqs)
                return {"entity_f1": m["entity_f1"], "token_accuracy": m["token_accuracy"]}
            m = pos_metrics(true_seqs, pred_seqs, td.labels)
            return {"token_accuracy": m["token_accuracy"], "macro_f1": m["macro_f1"]}

    out_dir = MODELS / task / run_id
    args = TrainingArguments(
        output_dir=str(out_dir), overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=eval_batch_size,
        learning_rate=head_lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear", eval_strategy="epoch", save_strategy="no",
        logging_strategy="steps", logging_steps=50, report_to=[], seed=seed,
        data_seed=seed, bf16=(device.type in {"mps", "cuda"}), dataloader_num_workers=0,
        disable_tqdm=False, label_names=(["start_positions", "end_positions"]
                                         if td.kind == "qa" else None),
    )

    timer = EpochTimer()
    TrainerCls = QATrainer if td.kind == "qa" else TwoGroupTrainer
    trainer = TrainerCls(model=model, args=args, train_dataset=train_ds,
                         eval_dataset=eval_ds, data_collator=collator,
                         processing_class=tokenizer, compute_metrics=compute_metrics,
                         callbacks=[timer])
    trainer.head_lr, trainer.body_lr = head_lr, body_lr

    if td.kind == "qa":
        trainer.pack = packs["dev"][:2]
        trainer.compute_metrics = trainer.score

    with Stopwatch() as sw_train:
        trainer.train()

    # -------------------- final evaluation on the held-out test split ---------------- #
    with Stopwatch() as sw_eval:
        if td.kind == "qa":
            examples, feats, model_in = packs["test"]
            trainer.pack = (examples, feats)
            trainer.predict(model_in, metric_key_prefix="test")
            preds = trainer.last_predictions
            metrics = squad_metrics(preds, examples)
            metrics["sample_predictions"] = [
                {"question": examples[i]["question"], "gold": examples[i]["answers"]["text"][0],
                 "predicted": preds[examples[i]["id"]]} for i in range(5)]
        else:
            preds = trainer.predict(tok["test"])
            logits, labels = preds.predictions, preds.label_ids
            if td.kind == "sequence":
                metrics = sequence_metrics(labels, np.argmax(logits, axis=-1), td.labels)
            else:
                true_seqs, pred_seqs = decode_token_predictions(logits, labels, td.labels)
                metrics = (ner_metrics(true_seqs, pred_seqs) if task == "ner"
                           else pos_metrics(true_seqs, pred_seqs, td.labels))

    if save_model:
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    rec = RunRecord(
        task=task, run_id=run_id, method=method, model_name=model_name, head=head,
        hyperparams={"epochs": epochs, "batch_size": batch_size, "head_lr": head_lr,
                     "body_lr": body_lr if method != "frozen" else None,
                     "weight_decay": weight_decay, "warmup_ratio": warmup_ratio,
                     "max_length": MAX_LEN[task], "bf16": args.bf16,
                     "scheduler": "linear with 10% warmup"},
        params=params, metrics=metrics,
        train_log=[d for d in trainer.state.log_history],
        epoch_times=timer.epochs, train_seconds=sw_train.seconds, eval_seconds=sw_eval.seconds,
        dataset=dataset_card(td), seed=seed, notes=notes)
    rec.save()
    return rec
