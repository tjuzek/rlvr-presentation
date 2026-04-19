# RLVR Presentation

Academic presentation on **Reinforcement Learning from Verifiable Rewards (RLVR)** for the [SC-AI Seminar](https://sc-ai.net/) at Florida State University. Presented by Tom and Lan Li.

Based on the Tulu 3 paper (Allen AI, 2024) and the broader RLVR landscape — training LLMs using automated verification (code tests, math solvers, constraint checks) as reward signals instead of human preference feedback.

## Running

```bash
python app.py  # serves presentation at localhost:8000
```

## Structure

- `slides/` — 23 markdown slide files (01-title through 23-references), read on each page load
- `demo/` — the four RLVR fine-tuning runs the talk references (`code-rlvr/`, `math-rlvr/`, `gemma-rlvr/`, `gemma-rlvr-v2/`) plus the unified results report at `demo/results/rlvr_demo_report.html`
- `demo/archive/` — superseded April-2026 proof-of-concept stub (GSM8K + PPO, single file)
- `static/css/`, `static/js/`, `static/img/` — theme, Plotly charts, paper figures
- `data/` — RLVR training datasets (GSM, MATH, IFEval) — gitignored, reproducible via HF datasets
- `articles/` — reference PDFs (Tulu paper, Allen AI blog, etc.) — gitignored
- `references.md` — full reading list (23 papers)
- `presentation.pdf`, `demo/results/rlvr_demo_report.pdf` — shareable PDF exports (regenerable via `make_slides_pdf.sh` and `demo/make_results_pdf.sh`)

## Design

Dark-mode reveal.js presentation with a "Apple keynote meets NeurIPS" aesthetic. The `ux-academic-reviewer` agent is configured for design review work.

## Slides

Markdown files in `slides/` with optional YAML frontmatter for slide classes and backgrounds. Edit them directly — the server reads fresh on each request.
