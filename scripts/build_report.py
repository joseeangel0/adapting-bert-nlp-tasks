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
        f"<tr><td>{'<b>' if r is best else ''}{r['method']}{'</b>' if r is best else ''}</td>"
        f"<td class='n'>{r['trainable']:,}</td><td class='n'>{r['trainable_pct']}%</td>"
        f"<td class='n {'best' if r is best else ''}'>{fmt(r['headline'])}</td>"
        f"<td class='n'>{fmt(r['secondary'])}</td><td class='n'>{r['minutes']:.1f}</td></tr>"
        for r in rs)
    return f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'


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
    body = []
    for r in rows():
        ets = [e for e in r["raw"].get("epoch_times", []) if "sec_per_step" in e]
        if not ets:
            continue
        sec_epoch = sum(e["seconds"] for e in ets) / len(ets)
        sec_step = sum(e["sec_per_step"] for e in ets) / len(ets)
        body.append(f"<tr><td>{TASK_TITLES[r['task']].split(' (')[0]}</td><td>{r['method']}</td>"
                    f"<td class='n'>{sec_epoch:.0f}</td><td class='n'>{sec_step:.3f}</td>"
                    f"<td class='n'>{r['minutes']:.1f}</td>"
                    f"<td class='n'>{r['raw'].get('eval_seconds', 0):.0f}</td></tr>")
    return f'<table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def table_qa_examples() -> str:
    best = best_per_task().get("qa")
    samples = (best or {}).get("raw", {}).get("metrics", {}).get("sample_predictions") or []
    if not samples:
        return ""
    body = "".join(f"<tr><td>{s['question']}</td><td>{s['gold']}</td><td>{s['predicted']}</td></tr>"
                   for s in samples)
    return ('<table class="keep"><thead><tr><th>Question</th><th>Gold answer</th>'
            f'<th>Predicted span</th></tr></thead><tbody>{body}</tbody></table>')


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
        # The gap is best-known-method minus best-frozen, not full minus frozen: on SQuAD the
        # best method we measured is partial fine-tuning, and quoting a missing run would be worse
        # than quoting the one that won.
        if f"VAL_{task}_frozen" in out:
            gap = b["headline"] - float(out[f"VAL_{task}_frozen"])
            out[f"VAL_{task}_gap"] = f"{gap:+.1f}"
        out[f"VAL_{task}_best_secondary"] = fmt(b["secondary"])
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

    out["VAL_pos_sk_macro"] = one("pos", "frozen_sk-logreg_firstsub_bert-base-uncased", "secondary")
    out["VAL_pos_partial_macro"] = one("pos", "partial_ft2_linear_bert-base-uncased", "secondary")
    out["VAL_pos_full_macro"] = one("pos", "full_ft_linear_bert-base-uncased", "secondary")

    cfg = json.loads((ROOT / "configs" / "report.json").read_text())
    out["TEAM"] = " &middot; ".join(n.replace(" ", "&nbsp;") for n in cfg["team"])
    out["REPO_URL"] = cfg.get("repo_url") or "local repository (not published)"
    env = data[0]["raw"]["env"]
    out["VAL_env"] = f"{env.get('chip', 'CPU')} - torch {env['torch']} on {env['device']}"
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
