"""
Unified RLVR results page — Overview + tabs for Runs 1–4.

Walks the four experiment directories, reads per-run JSON + metrics, and
emits a single standalone HTML at results/rlvr_demo_report.html.

Per-run reports are left in place — this is an additional top-level page
for people who want one link that tells the whole story.

Usage:
    python3 make_unified_report.py
    python3 make_unified_report.py --out path/to/out.html
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent

ACCENT = "#4d8eff"
SUCCESS = "#34d399"
DANGER = "#f87171"
WARN = "#fbbf24"
MUTED = "#a1a1aa"
PURPLE = "#a78bfa"

PLOT_LAYOUT_JS = """{
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(20,20,24,0.45)',
  font: { color: '#c7c7d1', family: 'JetBrains Mono, ui-monospace, monospace', size: 14 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.12)', tickfont: { size: 13 } },
  yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.12)', tickfont: { size: 13 } },
  margin: { l: 72, r: 24, t: 24, b: 56 },
  showlegend: true,
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#c7c7d1', size: 13 } },
}"""


# ── Experiment manifest ─────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "slug": "run-1",
        "dir": "code-rlvr",
        "tab": "Run 1 · Code (OLMo)",
        "eyebrow": "Run 1 · MBPP · OLMo-2-7B-Instruct",
        "title": "Code: baseline too low",
        "test_set": "MBPP held-out (257)",
        # code-rlvr predates the training_config.json convention —
        # supply the values it used so the recipe panel still renders.
        "fallback_config": {
            "model": "allenai/OLMo-2-1124-7B-Instruct",
            "dataset": "mbpp",
            "num_generations": 4,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_steps": 200,
            "batch_size": 4,
            "learning_rate": 5e-6,
            "lora_r": 16,
            "lora_alpha": 32,
            "kl_coeff": 0.05,
            "max_new_tokens": 512,
            "gradient_accumulation": 1,
        },
    },
    {
        "slug": "run-2",
        "dir": "math-rlvr",
        "tab": "Run 2 · Math (OLMo)",
        "eyebrow": "Run 2 · GSM8K · OLMo-2-7B-Instruct",
        "title": "Math on OLMo: baseline too high",
        "test_set": "openai/gsm8k test (1,319)",
    },
    {
        "slug": "run-3",
        "dir": "gemma-rlvr",
        "tab": "Run 3 · Math (Gemma)",
        "eyebrow": "Run 3 · GSM8K · Gemma-2-2B-IT · G=8",
        "title": "Variance fixed · update budget still capped",
        "test_set": "openai/gsm8k test (1,319)",
    },
    {
        "slug": "run-4",
        "dir": "gemma-rlvr-v2",
        "tab": "Run 4 · Math (Gemma v2)",
        "eyebrow": "Run 4 · GSM8K · Gemma-2-2B-IT · unlocked budget",
        "title": "The one that learns",
        "test_set": "openai/gsm8k test (1,319)",
    },
]


# ── Data loading ────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_experiment(exp: dict) -> dict:
    d = ROOT / exp["dir"]
    baseline = load_json(d / "results" / "baseline.json")
    post = load_json(d / "results" / "post_rlvr.json")
    metrics = load_jsonl(d / "output" / "metrics.jsonl")
    config = load_json(d / "output" / "training_config.json") or exp.get("fallback_config")
    return {
        **exp,
        "baseline": baseline,
        "post": post,
        "metrics": metrics,
        "config": config,
    }


# ── Small helpers ───────────────────────────────────────────────────────

def series(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and "step" in r:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def rolling_mean(ys: list[float], window: int = 5) -> list[float]:
    if len(ys) < window:
        return ys[:]
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        w = ys[lo:i + 1]
        out.append(sum(w) / len(w))
    return out


def fmt_pct(x: float | None) -> str:
    return f"{x:.2f}%" if x is not None else "—"


def fmt_delta(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}pp"


def fmt_kl(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "—"


def max_kl(metrics: list[dict]) -> float | None:
    xs = [r["kl"] for r in metrics if r.get("kl") is not None]
    return max(xs) if xs else None


def compute_flips(baseline: dict | None, post: dict | None) -> dict:
    if not baseline or not post:
        return {}
    b_map = {r["task_id"]: r["passed"] for r in baseline.get("results", [])}
    p_map = {r["task_id"]: r["passed"] for r in post.get("results", [])}
    common = set(b_map) & set(p_map)
    return {
        "fail_to_pass": sum(1 for t in common if not b_map[t] and p_map[t]),
        "pass_to_fail": sum(1 for t in common if b_map[t] and not p_map[t]),
        "still_pass": sum(1 for t in common if b_map[t] and p_map[t]),
        "still_fail": sum(1 for t in common if not b_map[t] and not p_map[t]),
        "total": len(common),
    }


# ── Per-run rendering ───────────────────────────────────────────────────

def render_recipe_panel(config: dict | None, baseline: dict | None) -> str:
    if not config:
        return ""
    rows = [
        ("Base model", str(config.get("model", "?"))),
        ("Generations per prompt (G)", str(config.get("num_generations", "?"))),
        ("Temperature", f"{config.get('temperature', '?')}"),
        ("Training steps", str(config.get("max_steps", "?"))),
        ("Learning rate", f"{config.get('learning_rate', '?')}"),
        ("KL coefficient β", f"{config.get('kl_coeff', '?')}"),
        ("LoRA r / α", f"{config.get('lora_r', '?')} / {config.get('lora_alpha', '?')}"),
        ("Max completion tokens", str(config.get("max_new_tokens", "?"))),
    ]
    if baseline and isinstance(baseline.get("accuracy"), (int, float)):
        b = baseline["accuracy"]
        g = config.get("num_generations")
        if isinstance(g, int) and g > 0:
            p = b / 100.0 if b > 1 else b
            allzero = ((1 - p) ** g) * 100
            allone = (p ** g) * 100
            band = 100 - allzero - allone
            rows.append(("Variance band (predicted)", f"{band:.1f}% mixed-reward"))
    cells = "\n".join(
        f'<div class="recipe-label">{escape(k)}</div>'
        f'<div class="recipe-value"><code>{escape(v)}</code></div>'
        for k, v in rows
    )
    return f'<h3>Recipe</h3><div class="recipe-grid">{cells}</div>'


def render_stat_cards(run: dict) -> str:
    b = run["baseline"]
    p = run["post"]
    b_acc = b["accuracy"] if b else None
    p_acc = p["accuracy"] if p else None
    delta = (p_acc - b_acc) if (b_acc is not None and p_acc is not None) else None
    n_steps = max((r.get("step", 0) for r in run["metrics"]), default=0)
    kl_max = max_kl(run["metrics"])
    cards = [
        ("Baseline pass@1", fmt_pct(b_acc), ACCENT),
        ("Post-RLVR pass@1", fmt_pct(p_acc), SUCCESS),
        ("Δ", fmt_delta(delta), SUCCESS if (delta or 0) >= 0 else DANGER),
        ("Training steps", f"{n_steps}", ACCENT),
        ("KL max", fmt_kl(kl_max), WARN),
    ]
    return "\n".join(
        f'<div class="stat"><div class="stat-value" style="color:{c}">{v}</div>'
        f'<div class="stat-label">{l}</div></div>'
        for l, v, c in cards
    )


def render_passat1_chart(run: dict, idp: str) -> str:
    b, p = run["baseline"], run["post"]
    if not b or not p:
        return ""
    bval, pval = b["accuracy"], p["accuracy"]
    data = (f"[{{x: ['Baseline', 'Post-RLVR'], y: [{bval:.2f}, {pval:.2f}], "
            f"type: 'bar', marker: {{color: ['{ACCENT}', '{SUCCESS}']}}, "
            f"text: ['{bval:.1f}%', '{pval:.1f}%'], textposition: 'outside', "
            f"textfont: {{color: '#e4e4e7', size: 18}}}}]")
    return f"""
<h3>pass@1 on {escape(run['test_set'])}</h3>
<div class="chart-frame"><div id="{idp}-passat1" class="chart"></div></div>
<script>
Plotly.newPlot('{idp}-passat1', {data},
  {{...PLOT, showlegend: false,
    yaxis: {{...PLOT.yaxis, title: 'pass@1 (%)', range: [0, 100]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_reward_chart(metrics: list[dict], idp: str) -> str:
    xs, ys = series(metrics, "reward")
    if len(xs) < 3:
        return ""
    ys_smooth = rolling_mean(ys, window=5)
    raw = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys)}, "
           f"type: 'scatter', mode: 'lines+markers', name: 'Per-log reward', "
           f"line: {{color: '{ACCENT}', width: 2}}, "
           f"marker: {{size: 5, color: '{ACCENT}'}}, opacity: 0.55}}")
    smooth = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys_smooth)}, "
              f"type: 'scatter', mode: 'lines', name: 'Rolling mean (5)', "
              f"line: {{color: '{SUCCESS}', width: 3}}}}")
    return f"""
<h3>Reward (fraction of completions with correct answer)</h3>
<div class="chart-frame"><div id="{idp}-reward" class="chart"></div></div>
<script>
Plotly.newPlot('{idp}-reward', [{raw}, {smooth}],
  {{...PLOT, xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Reward', range: [0, 1]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_variance_chart(metrics: list[dict], idp: str) -> str:
    xs, ys = series(metrics, "frac_reward_zero_std")
    if len(xs) < 3:
        return ""
    ys_pct = [y * 100 for y in ys]
    mean_pct = sum(ys_pct) / len(ys_pct)
    trace = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys_pct)}, "
             f"type: 'scatter', mode: 'lines+markers', name: 'Zero-variance', "
             f"line: {{color: '{DANGER}', width: 2}}, marker: {{size: 5}}, "
             f"fill: 'tozeroy', fillcolor: 'rgba(248,113,113,0.08)'}}")
    return f"""
<h3>Reward-variance diagnostic · <code>frac_reward_zero_std</code></h3>
<p class="chart-hint">
  Percent of GRPO groups where all G completions got the same reward.
  These groups have zero advantage and contribute no gradient.
  Run average: <b style="color:{DANGER}">{mean_pct:.1f}%</b>.
</p>
<div class="chart-frame"><div id="{idp}-variance" class="chart"></div></div>
<script>
Plotly.newPlot('{idp}-variance', [{trace}],
  {{...PLOT, showlegend: false,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Zero-variance groups (%)', range: [0, 100]}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_kl_chart(metrics: list[dict], idp: str) -> str:
    xs, ys = series(metrics, "kl")
    if len(xs) < 3:
        return ""
    trace = (f"{{x: {json.dumps(xs)}, y: {json.dumps(ys)}, "
             f"type: 'scatter', mode: 'lines+markers', name: 'KL', "
             f"line: {{color: '{WARN}', width: 2}}, marker: {{size: 5}}}}")
    return f"""
<h3>KL divergence from reference policy</h3>
<p class="chart-hint">Low = trained policy stays close to base. High = drift — watch for degradation.</p>
<div class="chart-frame"><div id="{idp}-kl" class="chart"></div></div>
<script>
Plotly.newPlot('{idp}-kl', [{trace}],
  {{...PLOT, showlegend: false,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'KL'}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_loss_chart(metrics: list[dict], idp: str) -> str:
    xs_l, ys_l = series(metrics, "loss")
    xs_g, ys_g = series(metrics, "grad_norm")
    if len(xs_l) < 3 and len(xs_g) < 3:
        return ""
    traces = []
    if len(xs_l) >= 3:
        traces.append(
            f"{{x: {json.dumps(xs_l)}, y: {json.dumps(ys_l)}, "
            f"type: 'scatter', mode: 'lines', name: 'Loss', "
            f"line: {{color: '{ACCENT}', width: 2}}, yaxis: 'y'}}"
        )
    if len(xs_g) >= 3:
        traces.append(
            f"{{x: {json.dumps(xs_g)}, y: {json.dumps(ys_g)}, "
            f"type: 'scatter', mode: 'lines', name: 'Grad norm', "
            f"line: {{color: '{DANGER}', width: 2, dash: 'dot'}}, yaxis: 'y2'}}"
        )
    return f"""
<h3>Policy loss and gradient norm</h3>
<div class="chart-frame"><div id="{idp}-loss" class="chart"></div></div>
<script>
Plotly.newPlot('{idp}-loss', [{", ".join(traces)}],
  {{...PLOT,
    xaxis: {{...PLOT.xaxis, title: 'Training step'}},
    yaxis: {{...PLOT.yaxis, title: 'Loss', side: 'left'}},
    yaxis2: {{gridcolor: 'rgba(255,255,255,0.06)', title: 'Grad norm',
             overlaying: 'y', side: 'right', color: '{DANGER}'}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


def render_flip_chart(flips: dict, idp: str) -> str:
    if not flips:
        return ""
    cats = ["Fixed (fail→pass)", "Regressed (pass→fail)", "Still passing", "Still failing"]
    vals = [flips["fail_to_pass"], flips["pass_to_fail"], flips["still_pass"], flips["still_fail"]]
    colors = [SUCCESS, DANGER, ACCENT, MUTED]
    trace = (f"{{x: {json.dumps(cats)}, y: {json.dumps(vals)}, "
             f"type: 'bar', marker: {{color: {json.dumps(colors)}}}, "
             f"text: {json.dumps(vals)}, textposition: 'outside', "
             f"textfont: {{color: '#e4e4e7', size: 15}}}}")
    return f"""
<h3>Per-problem flip analysis ({flips['total']} common problems)</h3>
<div class="chart-frame"><div id="{idp}-flip" class="chart"></div></div>
<script>
Plotly.newPlot('{idp}-flip', [{trace}],
  {{...PLOT, showlegend: false,
    yaxis: {{...PLOT.yaxis, title: 'Problems'}}}},
  {{displayModeBar: false, responsive: true}});
</script>"""


RUN_BLURBS = {
    "run-1": {
        "one_liner": "Baseline too low (2.7%) — sparse reward, no gradient.",
        "detail": (
            "MBPP is a code-completion benchmark where OLMo-2-7B-Instruct (not code-tuned) "
            "solves only 2.7% of held-out problems. With G=4, the fraction of groups where "
            "every completion fails is (1−0.027)^4 ≈ 89%. Most optimizer steps carry no signal. "
            "Δ = +0.4pp is within noise."
        ),
        "accent": DANGER,
    },
    "run-2": {
        "one_liner": "Baseline too high (82.6%) — groups saturate all-pass.",
        "detail": (
            "Same base model, same recipe, different task. GSM8K math is a sweet spot for "
            "OLMo-2's Tulu 3 fine-tune, so ~83% of completions pass — meaning most groups are "
            "all-pass, reward variance collapses, and the advantage term is zero most of the time. "
            "Δ = −0.5pp is within noise."
        ),
        "accent": DANGER,
    },
    "run-3": {
        "one_liner": "Variance fixed; update budget still capped. KL max 0.0011.",
        "detail": (
            "Swapped to Gemma-2-2B-IT (GSM8K baseline 58.5%, smack in the middle), bumped "
            "G=8 and temperature 1.0. Observed zero-variance share fell from ~80% (OLMo) to ~52% "
            "— about 2.4× more signal per step. But with lr=5e-6, β=0.05, 200 steps, the "
            "policy barely moved: KL max 0.0011. Δ = −0.3pp, still within noise."
        ),
        "accent": WARN,
    },
    "run-4": {
        "one_liner": "Same recipe as Run 3 with lr×4, β÷10, steps×2. +1.8pp — first signal outside noise.",
        "detail": (
            "Identical model, G, and reward setup as Run 3 — the only changes are to the "
            "policy update budget: lr=2e-5 (4×), β=0.005 (10× smaller), 400 steps (2×). "
            "KL max jumped to 0.032 — 29× Run 3's ceiling. pass@1 went 58.53% → 60.35%. "
            "120 problems fixed, 96 regressed, net +24."
        ),
        "accent": SUCCESS,
    },
}


def render_per_run_blurb(exp: dict) -> str:
    b = RUN_BLURBS[exp["slug"]]
    return f"""
<div class="run-blurb" style="border-left-color:{b['accent']}">
  <div class="run-blurb-lead"><b>{escape(b['one_liner'])}</b></div>
  <p>{escape(b['detail'])}</p>
</div>"""


def render_run_section(run: dict) -> str:
    idp = run["slug"]
    stat_cards = render_stat_cards(run)
    recipe = render_recipe_panel(run["config"], run["baseline"])
    blurb = render_per_run_blurb(run)
    passat1 = render_passat1_chart(run, idp)
    reward = render_reward_chart(run["metrics"], idp)
    variance = render_variance_chart(run["metrics"], idp)
    kl = render_kl_chart(run["metrics"], idp)
    loss = render_loss_chart(run["metrics"], idp)
    flip = render_flip_chart(compute_flips(run["baseline"], run["post"]), idp)

    return f"""
<section id="{idp}" class="tab-panel">
  <div class="eyebrow">{escape(run['eyebrow'])}</div>
  <h2 class="run-title">{escape(run['title'])}</h2>
  {blurb}
  <div class="stats">{stat_cards}</div>
  {recipe}
  {passat1}
  {reward}
  {variance}
  {kl}
  {loss}
  {flip}
</section>"""


# ── Overview section ────────────────────────────────────────────────────

def render_overview_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        b = r["baseline"]["accuracy"] if r["baseline"] else None
        p = r["post"]["accuracy"] if r["post"] else None
        d = (p - b) if (b is not None and p is not None) else None
        kl = max_kl(r["metrics"])
        cfg = r["config"] or {}
        task = "MBPP" if r["dir"] == "code-rlvr" else "GSM8K"
        model = (cfg.get("model", "") or "").split("/")[-1]
        g = cfg.get("num_generations", "?")
        is_learner = r["slug"] == "run-4"
        row_class = " class=\"learner\"" if is_learner else ""
        delta_color = SUCCESS if (d or 0) > 0.5 else (DANGER if (d or 0) < -0.5 else MUTED)
        rows.append(
            f'<tr{row_class}>'
            f'<td><a href="#{r["slug"]}">{escape(r["tab"])}</a></td>'
            f'<td>{task}</td>'
            f'<td><code>{escape(model)}</code></td>'
            f'<td class="num">{g}</td>'
            f'<td class="num">{fmt_pct(b)}</td>'
            f'<td class="num">{fmt_pct(p)}</td>'
            f'<td class="num" style="color:{delta_color};font-weight:600">{fmt_delta(d)}</td>'
            f'<td class="num">{fmt_kl(kl)}</td>'
            f'</tr>'
        )
    return f"""
<table class="summary">
  <thead><tr>
    <th>Run</th><th>Task</th><th>Base model</th><th>G</th>
    <th>Baseline pass@1</th><th>Post-RLVR pass@1</th><th>Δ</th><th>KL max</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def render_overview_diagnostic_cards() -> str:
    cards = []
    for slug in ("run-1", "run-2", "run-3", "run-4"):
        b = RUN_BLURBS[slug]
        tab = next(e["tab"] for e in EXPERIMENTS if e["slug"] == slug)
        cards.append(f"""
<a href="#{slug}" class="diagnostic-card" style="border-left-color:{b['accent']}">
  <div class="diagnostic-tab">{escape(tab)}</div>
  <div class="diagnostic-lead">{escape(b['one_liner'])}</div>
  <div class="diagnostic-more">Read the full run →</div>
</a>""")
    return '<div class="diagnostic-grid">' + "".join(cards) + "</div>"


def render_overview_section(runs: list[dict]) -> str:
    r4 = next(r for r in runs if r["slug"] == "run-4")
    flips = compute_flips(r4["baseline"], r4["post"])
    run4_delta = r4["post"]["accuracy"] - r4["baseline"]["accuracy"]

    return f"""
<section id="overview" class="tab-panel">
  <div class="eyebrow">RLVR demo · four runs · one narrative</div>
  <h2 class="hero-title">Four RLVR runs. One that learns.</h2>
  <p class="hero-sub">A walk through what had to go right.</p>

  <div class="bottom-line">
    <b>Bottom line:</b> RLVR training is a tough cookie. Modern instruct
    models are already strong — adding on top requires <b>three</b>
    variables to line up at once: baseline in the variance band, enough
    non-dead groups, and a policy update budget that actually lets the
    policy move. Runs 1–3 each missed one. Run 4 hit all three — and
    the KL ceiling is near the drift zone even so.
  </div>

  <h3>Headline</h3>
  {render_overview_table(runs)}

  <h3>The three constraints GRPO needs</h3>
  <div class="constraints">
    <div class="constraint">
      <div class="constraint-n">1</div>
      <div class="constraint-body">
        <b>Baseline in the variance sweet spot.</b> Not so low that every
        completion fails (Run 1: 2.7%), not so high that every completion
        passes (Run 2: 82.6%).
      </div>
    </div>
    <div class="constraint">
      <div class="constraint-n">2</div>
      <div class="constraint-body">
        <b>Enough non-dead groups.</b> For baseline <i>p</i> and group size
        <i>G</i>, the all-same share is <span class="math">p<sup>G</sup> + (1−p)<sup>G</sup></span>.
        Empirically per-prompt difficulty clusters outcomes, so the
        observed share is higher than the independence model predicts.
      </div>
    </div>
    <div class="constraint">
      <div class="constraint-n">3</div>
      <div class="constraint-body">
        <b>A policy update budget that lets the policy move.</b> Learning
        rate, KL coefficient β, step count. Runs 1–3 all used lr=5e-6,
        β=0.05, 200 steps — and the policy never meaningfully moved
        (Run 3's KL max was 0.0011). Reward variance was necessary but
        not sufficient.
      </div>
    </div>
  </div>

  <h3>Per-run diagnosis</h3>
  {render_overview_diagnostic_cards()}

  <h3>The takeaway</h3>
  <div class="callout">
    <p>
      <b>Run 4 validated the hypothesis.</b> Same model, same G, same reward
      setup as Run 3 — only the policy update budget changed
      (lr 4×, β ÷10, steps 2×). KL max went from 0.0011 → <b style="color:{WARN}">0.032</b>
      (29×). pass@1 went <b>58.53% → 60.35% (+{run4_delta:.2f}pp)</b>,
      the first delta outside noise across all four runs.
      <b>{flips['fail_to_pass']} fixed, {flips['pass_to_fail']} regressed</b>, net
      +{flips['fail_to_pass'] - flips['pass_to_fail']}.
    </p>
    <p>
      But KL max 0.032 is high enough that <b>drift is real</b> — 96 problems
      the base model used to solve are now broken. On a single-task benchmark
      that's still a net win; on a multi-task eval it would be the place to
      watch for collateral damage.
    </p>
  </div>

  <h3>Honest caveat</h3>
  <p class="caveat">
    Run 4's reward curve is noisy and drifts <i>slightly</i> down in the tail.
    Per-step reward is averaged over ~16 completions (2 prompts × G=8), so
    noise dominates the trend. The pass@1 number on 1,319 problems is the
    smoothed signal — that's where the +1.82pp lives. Worth flagging.
  </p>

  <h3>Read the runs</h3>
  <ul class="reading-order">
    <li><a href="#run-4">Run 4 — Gemma/GSM8K, update budget unlocked</a> — the only run that clearly learns.</li>
    <li><a href="#run-3">Run 3 — Gemma/GSM8K, variance band</a> — the setup that Run 4 builds on.</li>
    <li><a href="#run-2">Run 2 — OLMo/GSM8K</a> — baseline too high.</li>
    <li><a href="#run-1">Run 1 — OLMo/MBPP</a> — the original talk demo; baseline too low.</li>
  </ul>
</section>"""


# ── Tabbed shell ────────────────────────────────────────────────────────

def render_page(runs: list[dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    tab_links = "\n".join(
        f'<a href="#{r["slug"]}" class="tab" data-target="{r["slug"]}">{escape(r["tab"])}</a>'
        for r in runs
    )
    run_sections = "\n".join(render_run_section(r) for r in runs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RLVR demo — four runs, one narrative</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg-base:      #09090b;
    --bg-primary:   #0c0c0f;
    --bg-secondary: #141418;
    --bg-card:      #19191f;
    --bg-elevated:  #1f1f27;
    --text-primary:   #f0f0f2;
    --text-secondary: #c7c7d1;
    --text-muted:     #8b8b9a;
    --text-faint:     #65657a;
    --accent:  {ACCENT};
    --success: {SUCCESS};
    --danger:  {DANGER};
    --warn:    {WARN};
    --purple:  {PURPLE};
    --border:       rgba(255, 255, 255, 0.07);
    --border-light: rgba(255, 255, 255, 0.12);
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', 'Consolas', ui-monospace, monospace;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-card: 0 4px 16px -2px rgba(0, 0, 0, 0.35), 0 1px 0 0 rgba(255, 255, 255, 0.03) inset;
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg-base); }}
  body {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    font-size: 19px;
    line-height: 1.6;
    min-height: 100vh;
    background-color: var(--bg-base);
    background-image:
      radial-gradient(ellipse 70% 45% at 50% 0%, rgba(77, 142, 255, 0.055) 0%, transparent 65%),
      radial-gradient(ellipse 50% 35% at 85% 100%, rgba(167, 139, 250, 0.035) 0%, transparent 60%);
    background-attachment: fixed;
    font-feature-settings: 'ss01' 1, 'cv11' 1;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}

  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 2.5rem 4rem 4rem;
  }}

  /* ── Header ─────────────────────────────────────────────── */
  .report-header {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: baseline;
    gap: 2rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .report-header .eyebrow {{ margin-bottom: 0.4rem; }}
  .report-header h1 {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    background: linear-gradient(135deg, {ACCENT} 0%, {PURPLE} 55%, {ACCENT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .report-header .subtitle {{
    color: var(--text-muted);
    font-size: 1rem;
    font-family: var(--font-mono);
    margin: 0;
  }}
  .render-meta {{
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--text-faint);
    text-align: right;
    white-space: nowrap;
  }}

  /* ── Tabs ─────────────────────────────────────────────── */
  .tab-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0 0 2rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--bg-base);
    z-index: 10;
    padding-top: 1rem;
  }}
  .tab {{
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted);
    text-decoration: none;
    padding: 0.55rem 1rem;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    transition: all 150ms ease;
    letter-spacing: 0.01em;
  }}
  .tab:hover {{ color: var(--text-primary); background: var(--bg-card); }}
  .tab.active {{
    color: var(--text-primary);
    background: var(--bg-elevated);
    border-color: var(--border-light);
    box-shadow: 0 0 0 1px var(--accent) inset, 0 2px 6px rgba(77, 142, 255, 0.12);
  }}

  /* ── Tab panels ───────────────────────────────────────── */
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  .eyebrow {{
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }}
  h2, .hero-title, .run-title {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 0.75rem 0;
  }}
  .hero-title {{
    background: linear-gradient(135deg, {ACCENT} 0%, {PURPLE} 55%, {ACCENT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .hero-sub {{
    font-family: var(--font-mono);
    color: var(--text-muted);
    font-size: 1.05rem;
    margin: 0 0 1.75rem 0;
  }}

  h3 {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.008em;
    margin: 2.5rem 0 0.9rem 0;
    padding-bottom: 0.5rem;
    background-image: linear-gradient(to right, var(--accent), transparent 35%);
    background-size: 100% 2px;
    background-position: bottom left;
    background-repeat: no-repeat;
  }}

  p {{ margin: 0.5rem 0 0.9rem; color: var(--text-secondary); }}
  b {{ color: var(--text-primary); font-weight: 600; }}
  i {{ color: var(--purple); font-style: italic; }}

  /* ── Bottom-line lede ───────────────────────────────── */
  .bottom-line {{
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(77, 142, 255, 0.05) 100%);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 0 0 2.5rem 0;
    font-size: 1.1rem;
    line-height: 1.65;
    color: var(--text-secondary);
    box-shadow: var(--shadow-card);
  }}

  /* ── Summary table ──────────────────────────────────── */
  table.summary {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0 2rem 0;
    font-size: 0.98rem;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-card);
  }}
  table.summary th {{
    font-family: var(--font-mono);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-faint);
    text-align: left;
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg-base);
  }}
  table.summary td {{
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
  }}
  table.summary td.num {{
    font-family: var(--font-mono);
    text-align: right;
    font-feature-settings: 'tnum' 1, 'zero' 1;
  }}
  table.summary td code {{
    background: rgba(77, 142, 255, 0.08);
    color: var(--accent);
    padding: 0.12rem 0.45rem;
    border-radius: var(--radius-sm);
    font-size: 0.88rem;
  }}
  table.summary tr:last-child td {{ border-bottom: none; }}
  table.summary tr.learner td {{
    background: rgba(52, 211, 153, 0.05);
    font-weight: 600;
    color: var(--text-primary);
  }}
  table.summary td a {{
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px solid transparent;
  }}
  table.summary td a:hover {{ border-bottom-color: var(--accent); }}

  /* ── Three constraints ──────────────────────────────── */
  .constraints {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
    margin: 0.75rem 0 2rem;
  }}
  .constraint {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1.25rem;
    align-items: start;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.4rem;
    box-shadow: var(--shadow-card);
  }}
  .constraint-n {{
    font-family: var(--font-mono);
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
    padding-top: 0.1rem;
    min-width: 1.5rem;
  }}
  .constraint-body {{ color: var(--text-secondary); font-size: 1rem; line-height: 1.55; }}
  .math {{
    font-family: var(--font-mono);
    background: var(--bg-base);
    padding: 0.1rem 0.45rem;
    border-radius: var(--radius-sm);
    font-size: 0.92em;
    border: 1px solid var(--border);
    color: var(--text-primary);
  }}

  /* ── Diagnostic cards ───────────────────────────────── */
  .diagnostic-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin: 0.75rem 0 2rem;
  }}
  .diagnostic-card {{
    display: block;
    text-decoration: none;
    color: inherit;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.35rem;
    box-shadow: var(--shadow-card);
    transition: transform 150ms ease, box-shadow 150ms ease;
  }}
  .diagnostic-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px -6px rgba(0,0,0,0.5), 0 1px 0 0 rgba(255,255,255,0.04) inset;
  }}
  .diagnostic-tab {{
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.11em;
    margin-bottom: 0.5rem;
  }}
  .diagnostic-lead {{
    color: var(--text-primary);
    font-size: 1.02rem;
    font-weight: 500;
    line-height: 1.45;
    margin-bottom: 0.6rem;
  }}
  .diagnostic-more {{
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--accent);
  }}

  /* ── Callout ────────────────────────────────────────── */
  .callout {{
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(251, 191, 36, 0.04) 100%);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--warn);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
    margin: 0.75rem 0 2rem;
    font-size: 1.05rem;
    line-height: 1.65;
    box-shadow: var(--shadow-card);
  }}
  .callout p {{ margin: 0.45rem 0; color: var(--text-secondary); }}
  .callout p:first-child {{ margin-top: 0; }}
  .callout p:last-child {{ margin-bottom: 0; }}

  .caveat {{
    font-size: 1rem;
    color: var(--text-muted);
    border-left: 2px solid var(--border-light);
    padding-left: 1rem;
    margin: 0.5rem 0 2rem;
  }}
  .reading-order {{
    margin: 0.75rem 0 2rem;
    padding-left: 1.25rem;
    color: var(--text-secondary);
  }}
  .reading-order li {{ margin: 0.4rem 0; }}
  .reading-order a {{ color: var(--accent); text-decoration: none; }}
  .reading-order a:hover {{ text-decoration: underline; }}

  /* ── Per-run panels ─────────────────────────────────── */
  .run-blurb {{
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(77, 142, 255, 0.04) 100%);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.4rem;
    margin: 0.25rem 0 1.75rem;
    color: var(--text-secondary);
    box-shadow: var(--shadow-card);
  }}
  .run-blurb-lead {{ font-size: 1.06rem; margin-bottom: 0.5rem; }}
  .run-blurb p {{ margin: 0; font-size: 0.98rem; line-height: 1.6; }}

  /* ── Stat cards ─────────────────────────────────────── */
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.8rem;
    margin: 0.5rem 0 2rem 0;
  }}
  .stat {{
    position: relative;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.35rem;
    box-shadow: var(--shadow-card);
    overflow: hidden;
  }}
  .stat::before {{
    content: '';
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.18), transparent);
  }}
  .stat-value {{
    font-family: var(--font-mono);
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-feature-settings: 'tnum' 1, 'zero' 1;
  }}
  .stat-label {{
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.45rem;
  }}

  /* ── Recipe panel ───────────────────────────────────── */
  .recipe-grid {{
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.45rem 1.25rem;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.15rem 1.4rem;
    margin: 0.5rem 0 2rem;
    box-shadow: var(--shadow-card);
    font-size: 0.92rem;
  }}
  .recipe-label {{
    font-family: var(--font-mono);
    color: var(--text-faint);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    align-self: center;
  }}
  .recipe-value {{ color: var(--text-primary); align-self: center; }}
  .recipe-value code {{
    background: rgba(77, 142, 255, 0.08);
    color: var(--accent);
    padding: 0.1rem 0.45rem;
    border-radius: var(--radius-sm);
    font-size: 0.88rem;
  }}

  /* ── Charts ─────────────────────────────────────────── */
  .chart-frame {{
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.15rem 0.65rem;
    margin: 0.5rem 0 1.5rem;
    box-shadow: var(--shadow-card);
  }}
  .chart {{ width: 100%; max-width: 100%; height: 380px; margin: 0; }}
  .chart-hint {{
    color: var(--text-muted);
    font-size: 0.94rem;
    margin: -0.4rem 0 0.6rem;
  }}

  /* ── Footer ─────────────────────────────────────────── */
  footer {{
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.02em;
  }}
  footer a {{ color: var(--text-muted); text-decoration: underline; }}

  @media (max-width: 1100px) {{
    .container {{ padding: 2rem 1.5rem 3rem; }}
    .report-header {{ grid-template-columns: 1fr; }}
    .render-meta {{ text-align: left; }}
    h2, .hero-title, .run-title {{ font-size: 1.8rem; }}
  }}

  /* ── Print mode (activated by ?print=1) ─────────────────── */
  body.print-mode .tab-bar {{ display: none !important; }}
  body.print-mode .tab-panel {{
    display: block !important;
    page-break-before: always;
  }}
  body.print-mode #overview {{ page-break-before: auto; }}
  body.print-mode .container {{ max-width: 100%; padding: 1.5rem 2rem; }}
  body.print-mode .tab-panel {{ break-inside: avoid-page; }}
  @media print {{
    body.print-mode {{ background: var(--bg-base) !important; }}
    body.print-mode .chart-frame {{ break-inside: avoid; }}
    body.print-mode .stat,
    body.print-mode .constraint,
    body.print-mode .diagnostic-card,
    body.print-mode .run-blurb,
    body.print-mode .callout,
    body.print-mode .bottom-line {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">

<header class="report-header">
  <div>
    <div class="eyebrow">rlvr · demo · unified report</div>
    <h1>RLVR on small instruct models — four runs</h1>
    <div class="subtitle">Companion to the SC-AI Seminar talk on RLVR (April 2026) · Tommie &amp; Lan Li</div>
  </div>
  <div class="render-meta">Rendered {ts}</div>
</header>

<nav class="tab-bar">
  <a href="#overview" class="tab" data-target="overview">Overview</a>
  {tab_links}
</nav>

<script>
const PLOT = {PLOT_LAYOUT_JS};
</script>

{render_overview_section(runs)}
{run_sections}

<footer>
  Generated by <code>make_unified_report.py</code> ·
  <a href="https://github.com/tjuzek/rlvr-presentation/tree/main/demo">github.com/tjuzek/rlvr-presentation/tree/main/demo</a>
</footer>

</div>

<script>
(function() {{
  const params = new URLSearchParams(location.search);
  const isPrint = params.has('print');

  if (isPrint) {{
    document.body.classList.add('print-mode');
    // All panels now visible via CSS — resize every Plotly chart on the page.
    function resizeAll() {{
      document.querySelectorAll('.js-plotly-plot').forEach(el => {{
        try {{ Plotly.Plots.resize(el); }} catch (e) {{}}
      }});
    }}
    window.addEventListener('load', () => {{
      resizeAll();
      // Second pass after layout settles — Plotly sometimes needs one.
      setTimeout(resizeAll, 300);
    }});
    return;
  }}

  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');

  function show(slug) {{
    if (!slug || !document.getElementById(slug)) slug = 'overview';
    tabs.forEach(t => t.classList.toggle('active', t.dataset.target === slug));
    panels.forEach(p => p.classList.toggle('active', p.id === slug));
    // Plotly charts rendered in hidden panels have 0 dims; resize on display.
    const panel = document.getElementById(slug);
    if (panel) {{
      panel.querySelectorAll('.js-plotly-plot').forEach(el => {{
        try {{ Plotly.Plots.resize(el); }} catch (e) {{}}
      }});
    }}
    // Keep the URL hash in sync without re-triggering hashchange.
    if (location.hash.slice(1) !== slug) {{
      history.replaceState(null, '', '#' + slug);
    }}
  }}

  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
  window.addEventListener('load', () => show(location.hash.slice(1) || 'overview'));
  tabs.forEach(t => t.addEventListener('click', ev => {{
    ev.preventDefault();
    show(t.dataset.target);
  }}));
}})();
</script>
</body>
</html>
"""


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate unified RLVR results page")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "rlvr_demo_report.html")
    args = parser.parse_args()

    runs = [load_experiment(e) for e in EXPERIMENTS]

    # Summary to stderr so the user sees what was found.
    for r in runs:
        b = r["baseline"]["accuracy"] if r["baseline"] else None
        p = r["post"]["accuracy"] if r["post"] else None
        d = (p - b) if (b is not None and p is not None) else None
        print(f"  {r['slug']:7}  baseline={fmt_pct(b):>6}  post={fmt_pct(p):>6}  Δ={fmt_delta(d):>8}  "
              f"steps={max((m.get('step', 0) for m in r['metrics']), default=0):>4}  "
              f"kl_max={fmt_kl(max_kl(r['metrics']))}")

    html = render_page(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"\nUnified report written to: {args.out}")


if __name__ == "__main__":
    main()
