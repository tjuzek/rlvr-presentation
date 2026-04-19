# RLVR — Reinforcement Learning from Verifiable Rewards

A presentation for the [SC-AI Seminar](https://sc-ai.net/) at Florida State
University. Presented by **Tom** and **Lan Li** (April 2026).

Based on [Tulu 3](https://arxiv.org/abs/2411.15124) (Allen AI, 2024) and the
broader RLVR landscape — training LLMs with automated verification (code
tests, math solvers, constraint checks) as the reward signal.

## Companion repo — the experiments

The four RLVR fine-tuning runs referenced in the talk live in a separate
repo, self-contained and reproducible:

**→ [github.com/tjuzek/rlvr-demo](https://github.com/tjuzek/rlvr-demo)**

| Run | Task · Model | Δ pass@1 |
|---|---|---:|
| 1 | MBPP · OLMo-2-7B-Instruct | +0.4pp |
| 2 | GSM8K · OLMo-2-7B-Instruct | −0.5pp |
| 3 | GSM8K · Gemma-2-2B-IT | −0.3pp |
| 4 | GSM8K · Gemma-2-2B-IT (unlocked update budget) | **+1.8pp** |

See the [unified results report](https://github.com/tjuzek/rlvr-demo/blob/main/results/rlvr_demo_report.html)
for the full walkthrough of what had to go right.

## Running the presentation

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

## Reveal.js shortcuts

| Key | Action |
|-----|--------|
| Arrow keys | Navigate slides |
| `o` | Slide overview |
| `s` | Speaker notes |
| `f` | Fullscreen |
| `Esc` | Exit fullscreen/overview |

## Editing slides

Markdown files in `slides/`. The server reads them fresh on each page load.
Each file supports optional YAML frontmatter:

```markdown
---
class: title-slide
background: #1a1a2e
---

# Your slide content here
```

## Structure

```
app.py              FastAPI server that renders slides
slides/             Markdown slide files
static/css/         Dark academic theme
static/js/          Plotly charts
static/img/         Paper figures
templates/          Reveal.js scaffold
references.md       Full reading list (23 papers)
grpo_pseudocode.py  GRPO advantage pseudocode referenced in one of the slides
```

The fine-tuning pipelines (verifiers, training configs, result JSON, per-run
HTML reports) are **not** in this repo — see the companion `rlvr-demo` above.

## Credits

- Primary source: Lambert et al., [Tulu 3 (Allen AI, 2024)](https://arxiv.org/abs/2411.15124).
- Presentation by Tom, with Lan Li.
- Built with [reveal.js](https://revealjs.com/) and [FastAPI](https://fastapi.tiangolo.com/).
- RLVR demo pipelines authored by Anthropic's Claude via Claude Code.
