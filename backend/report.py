"""Self-contained downloadable HTML research report (no external assets, prints cleanly to PDF)."""
from __future__ import annotations

import html
import json

from .engine import MODELS
from .features import ALL_FEATURES

LIMITATIONS = [
    "Trades are assumed to execute at the daily close with a fixed commission and slippage in basis points. "
    "Real fills, market impact, bid-ask spreads and liquidity limits can be worse.",
    "Short positions (long/short mode) ignore borrowing fees and short-sale constraints.",
    "Each experiment trains on a single asset. Survivorship bias applies if the asset was chosen because it still exists today.",
    "Hyperparameters are user-chosen, not tuned with nested cross-validation; trying many settings and keeping the best "
    "is itself a form of overfitting (multiple-testing bias).",
    "Sharpe confidence intervals assume independent daily returns; autocorrelation and fat tails widen the true uncertainty.",
    "Past performance in a backtest does not guarantee future results. This is research software, not investment advice.",
]


def _pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def _num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _e(x):
    return html.escape(str(x))


def _svg_lines(series: list[tuple[str, list, list, str]], w=860, h=260, pct_axis=False) -> str:
    """series: [(label, dates, values, color)]. Simple multi-line SVG chart."""
    allv = [v for _, _, vals, _ in series for v in vals if v is not None]
    if not allv:
        return ""
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi, lo = hi + 0.01, lo - 0.01
    pad_l, pad_r, pad_t, pad_b = 56, 12, 14, 26
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" style="max-width:{w}px">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = pad_t + ih - ih * i / 4
        lab = f"{v * 100:.0f}%" if pct_axis else f"{v:.2f}"
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
                   f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" fill="#6b7280">{lab}</text>')
    for label, dates, vals, color in series:
        n = len(vals)
        pts = " ".join(f"{pad_l + iw * i / max(1, n - 1):.1f},{pad_t + ih - ih * (v - lo) / (hi - lo):.1f}"
                       for i, v in enumerate(vals) if v is not None)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"/>')
    d0 = series[0][1]
    if d0:
        out.append(f'<text x="{pad_l}" y="{h - 6}" font-size="10" fill="#6b7280">{d0[0]}</text>'
                   f'<text x="{w - pad_r}" y="{h - 6}" font-size="10" text-anchor="end" fill="#6b7280">{d0[-1]}</text>')
    out.append("</svg>")
    legend = " ".join(f'<span style="color:{c}">&#9632;</span> {_e(l)}' for l, _, _, c in series)
    return f'<div class="chart">{"".join(out)}<div class="legend">{legend}</div></div>'


def build_report(exp: dict, project: dict | None) -> str:
    cfg, r = exp["config"], exp["results"]
    sp, pm, st, bh = r["split"], r["predictive"], r["strategy"], r["benchmark"]
    model_label = MODELS[cfg["model"]["type"]]["label"]
    params = ", ".join(f"{k}={'unlimited' if v is None else v}" for k, v in cfg["model"]["params"].items())
    rob = r["robustness"]
    leaky = [f for f, v in r["leakage"]["features"].items() if v["leaky"]]
    synthetic = str(r["dataset"].get("name", "")).upper().startswith(("DEMO_", "MESSY"))

    diag = "".join(f'<li class="{d["level"]}"><b>{_e(d["title"])}.</b> {_e(d["detail"])}</li>' for d in r["diagnostics"])
    feats = "".join(f"<li><code>{_e(f)}</code>: {_e(ALL_FEATURES[f][1])}</li>" for f in cfg["features"])

    def row3(label, key, fmt):
        return (f"<tr><td>{label}</td><td>{fmt(st['train'].get(key))}</td><td>{fmt(st['test'].get(key))}</td>"
                f"<td>{fmt(bh['test'].get(key))}</td></tr>")

    pred_rows = "".join(
        f"<tr><td>{lab}</td><td>{fmt(pm['train'][k])}</td><td>{fmt(pm['test'][k])}</td></tr>"
        for lab, k, fmt in [("Accuracy", "accuracy", _pct), ("Majority-class baseline", "majority_baseline_accuracy", _pct),
                            ("Edge vs baseline", "edge_vs_majority", _pct), ("Balanced accuracy", "balanced_accuracy", _pct),
                            ("ROC AUC", "roc_auc", lambda x: _num(x, 3)), ("Precision", "precision", _pct),
                            ("Recall", "recall", _pct), ("Log loss", "log_loss", lambda x: _num(x, 4))])
    strat_rows = "".join([row3("Total return", "total_return", _pct), row3("CAGR", "cagr", _pct),
                          row3("Return before costs", "gross_total_return", _pct), row3("Costs paid", "total_costs", _pct),
                          row3("Position changes", "n_trades", lambda x: "–" if x is None else str(x)),
                          row3("Time in market", "exposure", _pct)])
    risk_rows = "".join([row3("Sharpe ratio", "sharpe", _num), row3("Sharpe before costs", "gross_sharpe", _num),
                         row3("Sortino ratio", "sortino", _num), row3("Max drawdown", "max_drawdown", _pct),
                         row3("Calmar ratio", "calmar", _num), row3("Annual volatility", "annual_volatility", _pct)])
    ci = st["test"].get("sharpe_ci95") or [None, None]

    c, cb = r["curves"]["test"], r["curves"]["test_buy_hold"]
    eq_chart = _svg_lines([("Strategy (net of costs)", c["dates"], c["equity"], "#4f46e5"),
                           ("Buy and hold", cb["dates"], cb["equity"], "#9ca3af")])
    dd_chart = _svg_lines([("Strategy drawdown", c["dates"], c["drawdown"], "#dc2626"),
                           ("Buy-and-hold drawdown", cb["dates"], cb["drawdown"], "#9ca3af")], h=180, pct_axis=True)
    ctr = r["curves"]["train"]
    tr_chart = _svg_lines([("In-sample strategy", ctr["dates"], ctr["equity"], "#059669"),
                           ("Buy and hold", r["curves"]["train_buy_hold"]["dates"], r["curves"]["train_buy_hold"]["equity"], "#9ca3af")], h=200)

    wf = r.get("walk_forward", {})
    wf_html = "<p>Walk-forward evaluation was not enabled.</p>"
    if wf.get("folds"):
        rows = "".join(f"<tr><td>{f['fold']}</td><td>{f['train_start']} → {f['train_end']}</td><td>{f['test_start']} → {f['test_end']}</td>"
                       f"<td>{_pct(f['train_accuracy'])}</td><td>{_pct(f['accuracy'])}</td><td>{_num(f['strategy_sharpe'])}</td>"
                       f"<td>{_num(f['buy_hold_sharpe'])}</td><td>{_pct(f['max_drawdown'])}</td></tr>" for f in wf["folds"])
        s = wf["summary"]
        wf_html = (f"<p>{wf['window'].title()} training window, retrained before each fold. Mean Sharpe <b>{_num(s['mean_sharpe'])}</b> "
                   f"(std {_num(s['std_sharpe'])}); positive in {_pct(s['pct_folds_positive'], 0)} of folds; beat buy-and-hold in "
                   f"{_pct(s['pct_folds_beat_buy_hold'], 0)}.</p><table><tr><th>Fold</th><th>Train</th><th>Test</th><th>Train acc</th>"
                   f"<th>Test acc</th><th>Sharpe</th><th>B&amp;H Sharpe</th><th>Max DD</th></tr>{rows}</table>")
    elif wf.get("skipped"):
        wf_html = f"<p>Skipped: {_e(wf['skipped'])}</p>"

    reg_rows = "".join(f"<tr><td>{_e(g['regime'])}</td><td>{g['days']}</td><td>{_pct(g['accuracy'])}</td><td>{_num(g['strategy_sharpe'])}</td>"
                       f"<td>{_num(g['buy_hold_sharpe'])}</td></tr>" for g in r.get("regimes", []))
    gen_rows = "".join(f"<tr><td>{_e(g['name'])}</td><td>{g['start']} → {g['end']}</td><td>{_pct(g['accuracy'])}</td>"
                       f"<td>{_num(g['strategy_sharpe'])}</td><td>{_num(g['buy_hold_sharpe'])}</td></tr>" for g in r.get("generalization", []))
    fi_rows = "".join(f"<tr><td><code>{_e(f['feature'])}</code></td><td>{_pct(f['importance'])}</td><td>{_e(f.get('direction', ''))}</td></tr>"
                      for f in r.get("feature_importance", []))
    leak_rows = "".join(f"<tr><td><code>{_e(f)}</code></td><td class=\"{'bad' if v['leaky'] else 'good'}\">{'LEAKS FUTURE DATA' if v['leaky'] else 'Causal ✓'}</td></tr>"
                        for f, v in r["leakage"]["features"].items())
    rep = r["reproducibility"]
    env = ", ".join(f"{k} {v}" for k, v in rep["environment"].items())
    lims = list(LIMITATIONS)
    if synthetic:
        lims.insert(0, "The dataset is SYNTHETIC demo data generated by a regime-switching model, not real market prices.")
    lim_html = "".join(f"<li>{_e(x)}</li>" for x in lims)

    score_txt = ("DISQUALIFIED<span class='sub' style='display:block;font-size:12px'>" + _e("; ".join(rob.get("disqualified_reasons", [])))
                 + "</span>") if rob["disqualified"] else _num(rob["score"], 3)
    cfg_json = _e(json.dumps(exp["config"], indent=2, sort_keys=True))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignalForge report: {_e(exp['name'])}</title>
<style>
body{{font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111827;max-width:960px;margin:32px auto;padding:0 16px}}
h1{{font-size:24px;margin:0}} h2{{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
.sub{{color:#6b7280}} table{{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:13px}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #f3f4f6}} th{{background:#f9fafb;font-weight:600}}
td:not(:first-child),th:not(:first-child){{font-variant-numeric:tabular-nums}}
li.bad,td.bad{{color:#b91c1c}} li.good,td.good{{color:#047857}} li.warn{{color:#92400e}} li{{margin:4px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.kpi{{border:1px solid #e5e7eb;border-radius:8px;padding:10px}} .kpi b{{display:block;font-size:20px}}
.chart{{margin:6px 0 14px}} .legend{{font-size:12px;color:#374151}} pre{{background:#f9fafb;padding:10px;overflow:auto;font-size:12px}}
code{{font-size:12px}} @media print{{body{{margin:0}}}}
</style></head><body>
<h1>{_e(exp['name'])}</h1>
<div class="sub">SignalForge research report · Experiment #{exp['id']} v{exp.get('version', 1)} · Project: {_e((project or {}).get('name', ''))} · Generated {exp.get('finished_at', '')}</div>
<div class="kpis">
<div class="kpi">Robustness score<b>{score_txt}</b></div>
<div class="kpi">Test accuracy<b>{_pct(pm['test']['accuracy'])}</b><span class="sub">baseline {_pct(pm['test']['majority_baseline_accuracy'])}</span></div>
<div class="kpi">Test return (net)<b>{_pct(st['test']['total_return'])}</b><span class="sub">buy &amp; hold {_pct(bh['test']['total_return'])}</span></div>
<div class="kpi">Test Sharpe<b>{_num(st['test']['sharpe'])}</b><span class="sub">95% CI [{_num(ci[0])}, {_num(ci[1])}]</span></div>
<div class="kpi">Max drawdown<b>{_pct(st['test']['max_drawdown'])}</b><span class="sub">buy &amp; hold {_pct(bh['test']['max_drawdown'])}</span></div>
</div>

<h2>1. Key findings</h2><ul>{diag}</ul>

<h2>2. Assumptions</h2>
<table>
<tr><td>Dataset</td><td>{_e(r['dataset']['name'])} ({r['dataset']['rows']} rows, {r['dataset']['usable_rows']} usable after feature warm-up)</td></tr>
<tr><td>Target</td><td>1 if the close-to-close return over the next {cfg['target']['horizon']} trading day(s) is above {_pct(cfg['target']['threshold'], 2)}, else 0</td></tr>
<tr><td>Model</td><td>{model_label} ({_e(params) or 'default parameters'}), random seed {cfg['seed']}</td></tr>
<tr><td>Training period</td><td>{sp['train_start']} → {sp['train_end']} ({sp['train_rows']} days)</td></tr>
<tr><td>Test period (unseen)</td><td>{sp['test_start']} → {sp['test_end']} ({sp['test_rows']} days)</td></tr>
<tr><td>Trading rule</td><td>{'Long when' if cfg['signal']['mode'] == 'long_only' else 'Long when, otherwise short when not'} P(up) ≥ {cfg['signal']['prob_threshold']}{', otherwise in cash' if cfg['signal']['mode'] == 'long_only' else ''}</td></tr>
<tr><td>Transaction costs</td><td>{cfg['costs']['commission_bps']} bps commission + {cfg['costs']['slippage_bps']} bps slippage per unit of position change</td></tr>
</table>
<p><b>Features ({len(cfg['features'])}):</b></p><ul>{feats}</ul>

<h2>3. Evaluation methodology</h2>
<ul>
<li><b>Chronological split.</b> The model is trained only on the earlier period and evaluated on the later, unseen period. Rows are never shuffled.</li>
<li><b>Purging.</b> The last {sp['purged_rows']} training row(s) were removed because their labels look {cfg['target']['horizon']} day(s) ahead into the test period. Embargo: {sp['embargo_rows']} test row(s) skipped.</li>
<li><b>No preprocessing leakage.</b> Feature scaling is fitted inside the model pipeline on training rows only.</li>
<li><b>Look-ahead test.</b> {_e(r['leakage']['method'])}</li>
<li><b>Trade timing.</b> A prediction made at the close of day t sets the position held during day t+1. Costs are charged on every change in position, including entry and exit.</li>
<li><b>Benchmarks.</b> Predictive: always predicting the majority class. Strategy: buying and holding the same asset over the same period with the same entry cost.</li>
<li><b>Walk-forward.</b> The model is retrained and tested on successive later periods to check that results are stable over time, not one lucky split.</li>
<li><b>Robustness score.</b> Out-of-sample Sharpe − 0.5 × Sharpe instability − 3 × (train accuracy − test accuracy) + 0.5 × (Sharpe − buy-and-hold Sharpe, capped at ±1). Leaky runs are disqualified.</li>
</ul>

<h2>4. Results</h2>
<p class="sub">Three separate questions: does the model predict direction (accuracy), does trading on it make money (returns), and is that money worth the risk (risk-adjusted)?</p>
<h3>Predictive accuracy</h3><table><tr><th></th><th>Train (in-sample)</th><th>Test (unseen)</th></tr>{pred_rows}</table>
<h3>Strategy returns</h3><table><tr><th></th><th>Train</th><th>Test</th><th>Buy &amp; hold (test)</th></tr>{strat_rows}</table>
<h3>Risk-adjusted performance</h3><table><tr><th></th><th>Train</th><th>Test</th><th>Buy &amp; hold (test)</th></tr>{risk_rows}</table>
<h3>Test-period equity (start = 1.0)</h3>{eq_chart}
<h3>Test-period drawdown</h3>{dd_chart}
<h3>In-sample equity (for comparison; not evidence of skill)</h3>{tr_chart}

<h2>5. Robustness checks</h2>
<h3>Walk-forward evaluation</h3>{wf_html}
<h3>Market regimes (test period)</h3>
{f'<table><tr><th>Regime</th><th>Days</th><th>Accuracy</th><th>Strategy Sharpe</th><th>B&amp;H Sharpe</th></tr>{reg_rows}</table>' if reg_rows else '<p>Not enough days per regime.</p>'}
<h3>Generalization to unseen assets</h3>
{f'<table><tr><th>Asset</th><th>Period</th><th>Accuracy</th><th>Strategy Sharpe</th><th>B&amp;H Sharpe</th></tr>{gen_rows}</table>' if gen_rows else '<p>No other assets were selected.</p>'}
<h3>Feature importance</h3><table><tr><th>Feature</th><th>Share</th><th>Direction</th></tr>{fi_rows}</table>
<h3>Look-ahead leakage test</h3><table><tr><th>Feature</th><th>Result</th></tr>{leak_rows}</table>
{'<p class="bad"><b>Leakage found in ' + _e(', '.join(leaky)) + '. All results above are invalid.</b></p>' if leaky else ''}

<h2>6. Reproducibility</h2>
<table>
<tr><td>Config hash</td><td><code>{rep['config_hash']}</code></td></tr>
<tr><td>Dataset SHA-256</td><td><code>{rep['dataset_sha256']}</code></td></tr>
<tr><td>Result fingerprint</td><td><code>{rep['result_fingerprint']}</code></td></tr>
<tr><td>Random seed</td><td>{rep['seed']}</td></tr>
<tr><td>Environment</td><td>{_e(env)}</td></tr>
</table>
<details><summary>Full configuration (JSON)</summary><pre>{cfg_json}</pre></details>

<h2>7. Limitations</h2><ul>{lim_html}</ul>
</body></html>"""
