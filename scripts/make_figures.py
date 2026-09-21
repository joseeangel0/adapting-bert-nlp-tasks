#!/usr/bin/env python
"""Build every figure in the report from results/*/*.json.

Forms are chosen by the job the data does:
  ladder        - dot plot: position, not bar length, so a 90->97 gap stays honest
                  without truncating a bar axis.
  cost_benefit  - scatter on a log parameter axis: the whole point of the assignment
                  is what each rung costs for what it buys.
  curves        - lines over steps/epochs: change over time, for debugging the runs.
  ner_entities  - dot plot per entity type: where the methods actually differ.
  pos_confusion - sequential single-hue heatmap: magnitude, one hue light->dark.

Colors: categorical slots 1-3 of the validated palette, assigned to the three rungs of
the ladder in fixed order and never cycled. Every mark carries a direct label, which is
also the relief the aqua slot needs on a light surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report_data import (HEADLINE, RUNG_COLOR, RUNGS, TASK_ORDER, TASK_TITLES,  # noqa: E402
                             curve, rows)

OUT = ROOT / "report" / "figures"
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#85837d", "#e6e4dd"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})


def _clean(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", color=RULE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _legend(fig, used, **kw):
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                          color=RUNG_COLOR[r], label=RUNGS[r]) for r in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.6, **kw)


# --------------------------------------------------------------------------- #

def fig_ladder(data):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    used = []
    for ax, task in zip(axes.ravel(), TASK_ORDER):
        rs = [r for r in data if r["task"] == task and r["headline"] is not None]
        if not rs:
            ax.axis("off"); continue
        rs = rs[::-1]                                  # best rung at the top
        y = np.arange(len(rs))
        for i, r in enumerate(rs):
            c = RUNG_COLOR[r["rung"]]
            used.append(r["rung"])
            ax.plot([0, r["headline"]], [i, i], color=c, alpha=0.22, linewidth=2, zorder=1)
            ax.plot(r["headline"], i, "o", color=c, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
            ax.annotate(f"{r['headline']:.1f}", (r["headline"], i), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=INK, fontweight="bold")
        lo = min(r["headline"] for r in rs)
        hi = max(r["headline"] for r in rs)
        pad = max(4.0, (hi - lo) * 0.28)
        ax.set_xlim(max(0, lo - pad), min(101.5, hi + pad * 1.35))
        ax.set_yticks(y, [r["method"] for r in rs], fontsize=7.6)
        ax.set_ylim(-0.7, len(rs) - 0.3)
        ax.set_xlabel(HEADLINE[task][1] + " (%)")
        ax.set_title(TASK_TITLES[task], loc="left", color=INK, fontweight="bold", pad=7)
        _clean(ax)
    _legend(fig, list(dict.fromkeys(["frozen", "partial", "full"])))
    fig.tight_layout(h_pad=2.4, w_pad=3.0)
    fig.savefig(OUT / "ladder.png")
    plt.close(fig)


def fig_cost_benefit(data):
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
    for ax, task in zip(axes, TASK_ORDER):
        rs = [r for r in data if r["task"] == task and r["headline"] is not None]
        if not rs:
            ax.axis("off"); continue
        for r in rs:
            x = max(r["trainable"], 1)
            ax.plot(x, r["headline"], "o", color=RUNG_COLOR[r["rung"]], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        # connect the best run of each rung: the shape of the cost/benefit curve
        best = {}
        for r in rs:
            if r["rung"] not in best or r["headline"] > best[r["rung"]]["headline"]:
                best[r["rung"]] = r
        pts = sorted(((max(b["trainable"], 1), b["headline"]) for b in best.values()))
        if len(pts) > 1:
            ax.plot(*zip(*pts), color=MUTED, linewidth=1.2, alpha=0.55, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel(HEADLINE[task][1] + " (%)")
        ax.set_title(TASK_TITLES[task].split(" (")[0], loc="left", color=INK,
                     fontweight="bold", fontsize=9, pad=6)
        _clean(ax, xgrid=False)
    _legend(fig, ["frozen", "partial", "full"], bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "cost_benefit.png")
    plt.close(fig)


def fig_curves(data):
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.0))
    for col, task in enumerate(TASK_ORDER):
        rs = [r for r in data if r["task"] == task and r["raw"].get("train_log")]
        ax_loss, ax_score = axes[0][col], axes[1][col]
        key = {"agnews": "eval_accuracy", "ner": "eval_entity_f1",
               "pos": "eval_token_accuracy", "qa": "eval_f1"}[task]
        for r in rs:
            c = RUNG_COLOR[r["rung"]]
            pts = curve(r["raw"], "loss")
            if pts:
                xs, ys = zip(*pts)
                ax_loss.plot(xs, ys, color=c, linewidth=1.6, alpha=0.9)
            pts = curve(r["raw"], key)
            if pts:
                seen, dedup = set(), []
                for e, v in pts:
                    if e not in seen:
                        seen.add(e); dedup.append((e, v))
                xs, ys = zip(*dedup)
                scale = 100 if task != "qa" else 1
                ax_score.plot(xs, [y * scale for y in ys], color=c, linewidth=1.8,
                              marker="o", markersize=5, markeredgecolor=SURFACE,
                              markeredgewidth=1.1)
        ax_loss.set_title(TASK_TITLES[task].split(" (")[0], loc="left", color=INK,
                          fontweight="bold", fontsize=9, pad=6)
        ax_loss.set_ylabel("Training loss" if col == 0 else "")
        ax_loss.set_xlabel("Optimisation step")
        ax_score.set_ylabel(f"Dev {HEADLINE[task][1].lower()} (%)" if col == 0 else "")
        ax_score.set_xlabel("Epoch")
        for ax in (ax_loss, ax_score):
            _clean(ax, xgrid=False)
    _legend(fig, ["frozen", "partial", "full"], bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    fig.savefig(OUT / "curves.png")
    plt.close(fig)


def fig_ner_entities(data):
    rs = [r for r in data if r["task"] == "ner" and r["raw"]["metrics"].get("per_entity")]
    if not rs:
        return
    types = sorted({t for r in rs for t in r["raw"]["metrics"]["per_entity"]})
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for j, r in enumerate(rs):
        c = RUNG_COLOR[r["rung"]]
        for i, t in enumerate(types):
            v = r["raw"]["metrics"]["per_entity"].get(t, {}).get("f1-score", 0) * 100
            ax.plot(v, i + (j - (len(rs) - 1) / 2) * 0.17, "o", color=c, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            if j == len(rs) - 1 or j == 0:
                ax.annotate(f"{v:.0f}", (v, i + (j - (len(rs) - 1) / 2) * 0.17),
                            xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=7, color=INK2)
    ax.set_yticks(range(len(types)), types)
    ax.set_xlabel("Entity F1 (%)")
    ax.set_title("CoNLL-2003 test: F1 per entity type", loc="left", color=INK,
                 fontweight="bold", pad=7)
    ax.set_ylim(-0.6, len(types) - 0.4)
    _clean(ax)
    _legend(fig, list(dict.fromkeys(r["rung"] for r in rs)), bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "ner_entities.png")
    plt.close(fig)


def fig_pos_confusion(data):
    rs = [r for r in data if r["task"] == "pos" and r["raw"]["metrics"].get("confusion_matrix")]
    if not rs:
        return
    best = max(rs, key=lambda r: r["headline"])
    m = np.array(best["raw"]["metrics"]["confusion_matrix"], dtype=float)
    labels = best["raw"]["metrics"]["labels"]
    norm = m / np.clip(m.sum(1, keepdims=True), 1, None) * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if norm[i, j] >= 1:
                ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=5.8,
                        color="white" if norm[i, j] > 55 else INK2)
    ax.set_xlabel("Predicted tag"); ax.set_ylabel("Gold tag")
    ax.set_title(f"UD English-EWT test, row-normalised (%)\n{best['method']}",
                 loc="left", color=INK, fontweight="bold", fontsize=9, pad=8)
    ax.grid(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "pos_confusion.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = rows()
    if not data:
        print("no results yet", file=sys.stderr); return 1
    fig_ladder(data); fig_cost_benefit(data); fig_curves(data)
    fig_ner_entities(data); fig_pos_confusion(data)
    made = sorted(p.name for p in OUT.glob("*.png"))
    print(f"[figures] {len(made)} written to {OUT}: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
