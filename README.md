# RLVR — Reinforcement Learning from Verifiable Rewards

A presentation for the [SC-AI Seminar](https://sc-ai.net/) at Florida State
University. Presented by **Tom** and **Lan Li** (April 2026).

Based on [Tulu 3](https://arxiv.org/abs/2411.15124) (Allen AI, 2024) and the
broader RLVR landscape — training LLMs with automated verification (code
tests, math solvers, constraint checks) as the reward signal.

## Download

- **[Slides (PDF, 30 pages)](presentation.pdf)** — the full reveal.js deck.
- **[Results report (PDF, 24 pages)](demo/results/rlvr_demo_report.pdf)** — Overview + the four training runs.

Live versions: `python app.py` serves the reveal.js slides; open
[`demo/results/rlvr_demo_report.html`](demo/results/rlvr_demo_report.html)
for the interactive report with working Plotly charts.

## Experiments — [`demo/`](demo/)

The four RLVR fine-tuning runs referenced in the talk live in the
[`demo/`](demo/) subdirectory of this repo — self-contained and reproducible.

| Run | Task · Model | Δ pass@1 |
|---|---|---:|
| 1 | MBPP · OLMo-2-7B-Instruct | +0.4pp |
| 2 | GSM8K · OLMo-2-7B-Instruct | −0.5pp |
| 3 | GSM8K · Gemma-2-2B-IT | −0.3pp |
| 4 | GSM8K · Gemma-2-2B-IT (unlocked update budget) | **+1.8pp** |

See the [unified results report](demo/results/rlvr_demo_report.html)
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
demo/               Four RLVR fine-tuning runs + unified results report
make_slides_pdf.sh  Regenerates presentation.pdf (requires app.py running on :8000)
presentation.pdf    Downloadable slide deck (linked at top)
```

## Credits

- Primary source: Lambert et al., [Tulu 3 (Allen AI, 2024)](https://arxiv.org/abs/2411.15124).
- Presentation by Tom, with Lan Li.
- Built with [reveal.js](https://revealjs.com/) and [FastAPI](https://fastapi.tiangolo.com/).
- RLVR demo pipelines authored by Anthropic's Claude via Claude Code.
