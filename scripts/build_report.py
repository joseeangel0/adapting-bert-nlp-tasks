#!/usr/bin/env python
"""Render report/report.html from report/report_template.html + results/, then print a PDF.

Every number in the report comes from results/*/*.json through these generators, so the
prose and the tables can never drift apart. Placeholders in the template look like
{{TABLE_LADDER}} or {{VAL_agnews_best}}.

    python scripts/build_report.py            # html + pdf
    python scripts/build_report.py --no-pdf   # html only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report_data import (HEADLINE, RUNGS, SECONDARY, TASK_ORDER, TASK_TITLES,  # noqa: E402
                             best_per_task, rows)

REPORT = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def lower_metric(name: str) -> str:
    """Metric name mid-sentence: 'accuracy', 'entity F1' - never 'entity f1'."""
    return name if name == "F1" else name[0].lower() + name[1:]


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def env_label(r: dict) -> str:
    env = r["raw"]["env"]
    return f"{env.get('chip', 'CPU')} - torch {env['torch']} on {env['device']}"


@lru_cache(maxsize=None)
def main_env() -> str:
    """The environment most runs were trained on; anything else gets a dagger in the tables."""
    labels = [env_label(r) for r in rows()]
    return max(set(labels), key=labels.count)


def hw_mark(r: dict) -> str:
    return "<sup>†</sup>" if env_label(r) != main_env() else ""


def hw_note(rs: list[dict]) -> str:
    """Footnote for a table holding runs from more than one machine: their clocks don't compare."""
    other = sorted({env_label(r) for r in rs if hw_mark(r)})
    if not other:
        return ""
    return (f"<p class='small'>† trained on {', '.join(other)} instead of {main_env()}. Its scores "
            "compare with the other rows; its wall-clock times do not.</p>")


def table_matrix() -> str:
    """Tasks x methods: the comparison table the assignment asks for."""
    data = rows()
    rungs = ["frozen", "partial", "full"]
    head = ("<tr><th>Task</th><th>Metric</th>"
            + "".join(f'<th class="n">{RUNGS[r]}</th>' for r in rungs)
            + "<th>Delivered</th></tr>")
    body = []
    for task in TASK_ORDER:
        rs = [r for r in data if r["task"] == task and r["headline"] is not None]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["headline"])
        cells = []
        for rung in rungs:
            cand = [r for r in rs if r["rung"] == rung]
            if not cand:
                cells.append('<td class="n">—</td>'); continue
            b = max(cand, key=lambda r: r["headline"])
            cls = "n best" if b is best else "n"
            cells.append(f'<td class="{cls}">{fmt(b["headline"])}</td>')
        body.append(f"<tr><td>{TASK_TITLES[task]}</td><td>{HEADLINE[task][1]}</td>"
                    + "".join(cells) + f"<td>{best['method']}</td></tr>")
    return f'<table class="keep"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def table_runs(task: str) -> str:
    rs = [r for r in rows(task) if r["headline"] is not None]
    if not rs:
        return "<p class='small'>no runs recorded</p>"
    hname, sname = HEADLINE[task][1], SECONDARY[task][1]
    best = max(rs, key=lambda r: r["headline"])
    head = (f"<tr><th>Method</th><th class='n'>Trainable params</th><th class='n'>Share</th>"
            f"<th class='n'>{hname}</th><th class='n'>{sname}</th>"
            f"<th class='n'>Train (min)</th></tr>")
    body = "".join(
        f"<tr><td>{'<b>' if r is best else ''}{r['method']}{'</b>' if r is best else ''}{hw_mark(r)}</td>"
        f"<td class='n'>{r['trainable']:,}</td><td class='n'>{r['trainable_pct']}%</td>"
        f"<td class='n {'best' if r is best else ''}'>{fmt(r['headline'])}</td>"
        f"<td class='n'>{fmt(r['secondary'])}</td><td class='n'>{r['minutes']:.1f}</td></tr>"
        for r in rs)
    return f'<table><thead>{head}</thead><tbody>{body}</tbody></table>{hw_note(rs)}'


def table_delivered() -> str:
    best = best_per_task()
    head = ("<tr><th>Task</th><th>Delivered model</th><th>Method</th>"
            "<th class='n'>Score</th><th class='n'>Trainable</th></tr>")
    slug = {"agnews": "bert-base-uncased-agnews-topic", "ner": "bert-base-uncased-conll2003-ner",
            "pos": "bert-base-uncased-ud-ewt-pos", "qa": "bert-base-uncased-squad-qa"}
    body = "".join(
        f"<tr><td>{TASK_TITLES[t]}</td><td><code>{slug[t]}</code></td><td>{r['method']}</td>"
        f"<td class='n'><b>{fmt(r['headline'])}</b> {lower_metric(HEADLINE[t][1])}</td>"
        f"<td class='n'>{r['trainable']:,}</td></tr>"
        for t, r in sorted(best.items(), key=lambda kv: TASK_ORDER.index(kv[0])))
    return f'<table class="keep"><thead>{head}</thead><tbody>{body}</tbody></table>'


def table_datasets() -> str:
    head = ("<tr><th>Task</th><th>Dataset</th><th class='n'>Train</th><th class='n'>Dev</th>"
            "<th class='n'>Test</th><th class='n'>Labels</th><th>License</th></tr>")
    body = []
    seen = set()
    for r in rows():
        if r["task"] in seen:
            continue
        seen.add(r["task"])
        ds = r["raw"]["dataset"]
        s = ds["sizes"]
        body.append(
            f"<tr><td>{TASK_TITLES[r['task']].split(' (')[0]}</td><td><code>{ds['source']}</code></td>"
            f"<td class='n'>{s.get('train', 0):,}</td><td class='n'>{s.get('dev', 0):,}</td>"
            f"<td class='n'>{s.get('test', 0):,}</td>"
            f"<td class='n'>{ds['num_labels'] or 'span'}</td><td class='small'>{ds['license']}</td></tr>")
    order = {t: i for i, t in enumerate(TASK_ORDER)}
    return f'<table class="keep"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def table_timing() -> str:
    head = ("<tr><th>Task</th><th>Method</th><th class='n'>sec / epoch</th>"
            "<th class='n'>sec / step</th><th class='n'>Total train (min)</th>"
            "<th class='n'>Test eval (s)</th></tr>")
    body, timed = [], []
    for r in rows():
        ets = [e for e in r["raw"].get("epoch_times", []) if "sec_per_step" in e]
        if not ets:
            continue
        timed.append(r)
        sec_epoch = sum(e["seconds"] for e in ets) / len(ets)
        sec_step = sum(e["sec_per_step"] for e in ets) / len(ets)
        body.append(f"<tr><td>{TASK_TITLES[r['task']].split(' (')[0]}</td><td>{r['method']}{hw_mark(r)}</td>"
                    f"<td class='n'>{sec_epoch:.0f}</td><td class='n'>{sec_step:.3f}</td>"
                    f"<td class='n'>{r['minutes']:.1f}</td>"
                    f"<td class='n'>{r['raw'].get('eval_seconds', 0):.0f}</td></tr>")
    return f'<table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>{hw_note(timed)}'


def table_qa_examples() -> str:
    """The same questions answered by the best frozen run and by the delivered one, side by side:
    the prose in 6.4 describes the frozen model's mistakes, so they have to be on the page."""
    best = best_per_task().get("qa")
    frozen = [r for r in rows("qa") if r["rung"] == "frozen" and r["headline"] is not None]
    if not best or not frozen:
        return ""
    frozen = max(frozen, key=lambda r: r["headline"])
    delivered = best["raw"]["metrics"].get("sample_predictions") or []
    answered = {s["question"]: s["predicted"]
                for s in frozen["raw"]["metrics"].get("sample_predictions") or []}
    if not delivered:
        return ""
    body = "".join(f"<tr><td>{s['question']}</td><td>{s['gold']}</td>"
                   f"<td>{answered.get(s['question'], '—')}</td><td>{s['predicted']}</td></tr>"
                   for s in delivered)
    return ('<table class="keep"><thead><tr><th>Question</th><th>Gold answer</th>'
            f'<th>{frozen["method"].replace("Feature-based (", "Frozen (")}</th>'
            f'<th>Delivered ({best["method"]})</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def table_hub() -> str:
    """The four Hub repositories, with the account namespace if one is configured yet."""
    cfg = json.loads((ROOT / "configs" / "report.json").read_text())
    user = cfg.get("hf_user") or ""
    slug = {"agnews": "bert-base-uncased-agnews-topic", "ner": "bert-base-uncased-conll2003-ner",
            "pos": "bert-base-uncased-ud-ewt-pos", "qa": "bert-base-uncased-squad-qa"}
    pipeline = {"agnews": "text-classification", "ner": "token-classification",
                "pos": "token-classification", "qa": "question-answering"}
    best = best_per_task()
    head = ("<tr><th>Task</th><th>Repository</th><th><code>pipeline_tag</code></th>"
            "<th class='n'>Reported metric</th></tr>")
    body = []
    for t in TASK_ORDER:
        r = best.get(t)
        if not r:
            continue
        repo = f"{user}/{slug[t]}" if user else slug[t]
        cell = (f'<a href="https://huggingface.co/{repo}">{repo}</a>' if user
                else f"<code>{repo}</code>")
        body.append(f"<tr><td>{TASK_TITLES[t].split(' (')[0]}</td><td>{cell}</td>"
                    f"<td><code>{pipeline[t]}</code></td>"
                    f"<td class='n'>{HEADLINE[t][1]} {fmt(r['headline'])}</td></tr>")
    note = "" if user else ('<p class="small">The repository names above are the slugs created by '
                            '<code>scripts/push_to_hub.py --user &lt;account&gt;</code>; the account '
                            'namespace is filled in from <code>configs/report.json</code> when the '
                            'models are pushed.</p>')
    return f'<table class="keep"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>{note}'


def values() -> dict[str, str]:
    """Scalars the prose interpolates, so sentences never contradict the tables."""
    out: dict[str, str] = {}
    data = rows()
    best = best_per_task()
    for task in TASK_ORDER:
        rs = [r for r in data if r["task"] == task and r["headline"] is not None]
        if not rs:
            continue
        b = best[task]
        out[f"VAL_{task}_best"] = fmt(b["headline"])
        out[f"VAL_{task}_best_method"] = b["method"]
        out[f"VAL_{task}_metric"] = lower_metric(HEADLINE[task][1])
        for rung in ("frozen", "partial", "full"):
            cand = [r for r in rs if r["rung"] == rung]
            if cand:
                top = max(cand, key=lambda r: r["headline"])
                out[f"VAL_{task}_{rung}"] = fmt(top["headline"])
                out[f"VAL_{task}_{rung}_method"] = top["method"]
                out[f"VAL_{task}_{rung}_min"] = f"{top['minutes']:.1f}"
        # The gap is best-known-method minus best-frozen, not full minus frozen, so a task where
        # a cheaper rung wins (or the full run is missing) never quotes a number it doesn't have.
        if f"VAL_{task}_frozen" in out:
            gap = b["headline"] - float(out[f"VAL_{task}_frozen"])
            out[f"VAL_{task}_gap"] = f"{gap:+.1f}"
            out[f"VAL_{task}_gap_abs"] = f"{gap:.0f}"
        out[f"VAL_{task}_best_secondary"] = fmt(b["secondary"])
        out[f"VAL_{task}_best_trainable"] = f"{b['trainable'] / 1e6:.1f} M"
        if f"VAL_{task}_partial" in out and f"VAL_{task}_full" in out:
            out[f"VAL_{task}_full_over_partial"] = (
                f"{float(out[f'VAL_{task}_full']) - float(out[f'VAL_{task}_partial']):.1f}")
    if "VAL_qa_gap" in out and "VAL_agnews_gap" in out:
        out["VAL_gap_ratio"] = f"{float(out['VAL_qa_gap']) / float(out['VAL_agnews_gap']):.0f}"
    # Specific runs the prose names directly, looked up by run id so a sentence can never
    # quote a number that belongs to a different experiment.
    by_id = {(r["task"], r["run_id"]): r for r in data}

    def one(task: str, run_id: str, which: str = "headline") -> str:
        r = by_id.get((task, run_id))
        return fmt(r[which]) if r and r.get(which) is not None else "—"

    out["VAL_agnews_probe_linear"] = one("agnews", "frozen_linear_bert-base-uncased")
    out["VAL_agnews_probe_mlp"] = one("agnews", "frozen_mlp_bert-base-uncased")
    out["VAL_agnews_probe_pooler"] = one("agnews", "frozen_linear-pooler_bert-base-uncased")
    out["VAL_agnews_sk_cls"] = one("agnews", "frozen_sk-logreg_cls_bert-base-uncased")

    ner_sk = by_id.get(("ner", "frozen_sk-logreg_firstsub_bert-base-uncased"))
    ner_mlp = by_id.get(("ner", "frozen_mlp_bert-base-uncased"))
    if ner_sk and ner_mlp and ner_sk["headline"] and ner_mlp["headline"]:
        out["VAL_ner_probe_gain"] = f"{ner_mlp['headline'] - ner_sk['headline']:.1f}"

    a = by_id.get(("agnews", "frozen_linear_bert-base-uncased"))
    b_ = by_id.get(("agnews", "frozen_linear-pooler_bert-base-uncased"))
    if a and b_:
        out["VAL_agnews_pooler_gap"] = f"{b_['headline'] - a['headline']:.1f}"

    flat = by_id.get(("agnews", "partial_ft2_linear_bert-base-uncased"))
    if flat:
        ev = [e["eval_accuracy"] for e in flat["raw"]["train_log"] if "eval_accuracy" in e]
        out["VAL_flat_example"] = (f"{(ev[-1] - ev[-2]) * 100:+.2f} points" if len(ev) > 1 else "—")

    def last_epoch_gain(task: str, run_id: str, key: str) -> str:
        r = by_id.get((task, run_id))
        ev = [e[key] for e in r["raw"]["train_log"] if key in e] if r else []
        return f"{ev[-1] - ev[-2]:+.1f}" if len(ev) > 1 else "—"

    out["VAL_qa_partial_last_gain"] = last_epoch_gain("qa", "partial_ft4_linear_bert-base-uncased", "eval_f1")
    out["VAL_qa_full_last_gain"] = last_epoch_gain("qa", "full_ft_linear_bert-base-uncased", "eval_f1")
    qa_partial = by_id.get(("qa", "partial_ft4_linear_bert-base-uncased"))
    if qa_partial:
        out["VAL_qa_partial_spread"] = f"{qa_partial['headline'] - qa_partial['secondary']:.1f}"
    qa_best = best.get("qa")
    if qa_best:
        out["VAL_qa_best_spread"] = f"{qa_best['headline'] - qa_best['secondary']:.1f}"

    out["VAL_pos_sk_macro"] = one("pos", "frozen_sk-logreg_firstsub_bert-base-uncased", "secondary")
    out["VAL_pos_partial_macro"] = one("pos", "partial_ft2_linear_bert-base-uncased", "secondary")
    out["VAL_pos_full_macro"] = one("pos", "full_ft_linear_bert-base-uncased", "secondary")

    cfg = json.loads((ROOT / "configs" / "report.json").read_text())
    out["TEAM"] = " &middot; ".join(n.replace(" ", "&nbsp;") for n in cfg["team"])
    out["REPO_URL"] = cfg.get("repo_url") or "local repository (not published)"
    out["VAL_env"] = main_env()
    elsewhere = [r for r in data if hw_mark(r)]
    out["VAL_env_main_runs"] = str(len(data) - len(elsewhere))
    out["VAL_env_other"] = ", ".join(sorted({env_label(r) for r in elsewhere})) or "—"
    counts = {env: sum(env_label(r) == env for r in elsewhere) for env in {env_label(r) for r in elsewhere}}
    out["VAL_envs"] = main_env() + "".join(
        f" + {env} ({n} run{'s' if n != 1 else ''})" for env, n in sorted(counts.items()))
    out["VAL_total_runs"] = str(len(data))
    out["VAL_total_minutes"] = f"{sum(r['minutes'] for r in data):.0f}"
    return out


def render() -> str:
    template = (REPORT / "report_template.html").read_text()
    repl = {
        "{{TABLE_MATRIX}}": table_matrix(),
        "{{TABLE_DELIVERED}}": table_delivered(),
        "{{TABLE_DATASETS}}": table_datasets(),
        "{{TABLE_TIMING}}": table_timing(),
        "{{TABLE_QA_EXAMPLES}}": table_qa_examples(),
        "{{TABLE_HUB}}": table_hub(),
    }
    for task in TASK_ORDER:
        repl[f"{{{{TABLE_RUNS_{task}}}}}"] = table_runs(task)
    for k, v in values().items():
        repl["{{" + k + "}}"] = v

    html = template
    for k, v in repl.items():
        html = html.replace(k, v)
    leftovers = [line for line in html.splitlines() if "{{" in line]
    if leftovers:
        print(f"[warn] {len(leftovers)} unresolved placeholder line(s):", file=sys.stderr)
        for line in leftovers[:8]:
            print("   ", line.strip()[:120], file=sys.stderr)
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    html = render()
    out_html = REPORT / "report.html"
    out_html.write_text(html)
    print(f"[html] {out_html.relative_to(ROOT)} ({len(html):,} bytes)")

    if args.no_pdf:
        return 0
    pdf = REPORT / "U2T01_report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True,
                   capture_output=True)
    print(f"[pdf]  {pdf.relative_to(ROOT)} ({pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
