"""
RLVR Presentation Server

Run with: python app.py
Then open http://localhost:8000 in your browser.

Serves a reveal.js presentation with slides defined in Markdown files
under the slides/ directory. Edit the .md files to change content.
"""

import glob
import hashlib
import os
import re
from pathlib import Path

import markdown
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RLVR Presentation")

BASE_DIR = Path(__file__).parent
SLIDES_DIR = BASE_DIR / "slides"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def load_slides() -> list[dict]:
    """Load all slide markdown files in order, parse metadata and content."""
    slide_files = sorted(glob.glob(str(SLIDES_DIR / "*.md")))
    slides = []
    for fpath in slide_files:
        with open(fpath, "r") as f:
            raw = f.read()

        # Parse optional YAML-like frontmatter (---\nkey: val\n---\n)
        meta = {}
        content = raw
        fm_match = re.match(r"^---\n(.*?)\n---\n(.*)", raw, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            content = fm_match.group(2)

        # Protect math blocks from markdown processing.
        # KaTeX needs the raw LaTeX — but Python-Markdown mangles
        # underscores, backslashes, etc.  We swap math spans out for
        # unique placeholders, run Markdown, then restore them.
        math_store: dict[str, str] = {}

        def _stash_math(m: re.Match) -> str:
            raw = m.group(0)
            key = f"MATH_{hashlib.md5(raw.encode()).hexdigest()}"
            math_store[key] = raw
            return key

        # Block math first ($$...$$), then inline ($...$).
        content = re.sub(r"\$\$.+?\$\$", _stash_math, content, flags=re.DOTALL)
        content = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _stash_math, content)

        # Convert markdown to HTML
        html = markdown.markdown(
            content,
            extensions=[
                "fenced_code",
                "tables",
                "attr_list",
                "md_in_html",
            ],
        )

        # Restore math blocks (placeholders may be wrapped in <p> tags)
        for key, raw in math_store.items():
            html = html.replace(key, raw)

        slides.append(
            {
                "filename": os.path.basename(fpath),
                "meta": meta,
                "html": html,
            }
        )
    return slides


def build_presentation_html(slides: list[dict]) -> str:
    """Build the full reveal.js HTML from slides."""
    sections = []
    for slide in slides:
        bg = slide["meta"].get("background", "")
        classes = slide["meta"].get("class", "")
        bg_attr = f' data-background="{bg}"' if bg else ""
        cls_attr = f' class="{classes}"' if classes else ""
        # Support vertical sub-slides separated by <!--vsplit-->
        parts = slide["html"].split("<!--vsplit-->")
        if len(parts) > 1:
            inner = "\n".join(
                f"<section{bg_attr if i == 0 else ''}>{p}</section>"
                for i, p in enumerate(parts)
            )
            sections.append(f"<section>{inner}</section>")
        else:
            sections.append(
                f"<section{bg_attr}{cls_attr}>{slide['html']}</section>"
            )

    slides_html = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLVR &mdash; Reinforcement Learning from Verifiable Rewards</title>

    <!-- Favicon — diffusion grid -->
    <link rel="icon" type="image/svg+xml" href="/static/img/favicon.svg">

    <!-- Reveal.js core -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.css" id="theme">

    <!-- Code highlighting — use a darker, less saturated theme -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/highlight/monokai.css">

    <!-- Preconnect to Google Fonts for faster font loading -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

    <!-- Custom theme (must load after reveal theme to override) -->
    <link rel="stylesheet" href="/static/css/theme.css">

    <!-- Plotly for interactive charts -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

    <!-- Prevent FOUC: set background immediately -->
    <style>
        html, body {{ background: #09090b; }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            {slides_html}
        </div>
    </div>

    <!-- Custom navigation buttons -->
    <div class="slide-nav">
        <div class="slide-nav-group slide-nav-left">
            <button id="nav-fast-prev" class="slide-nav-btn" title="Back 5 slides" aria-label="Back 5 slides">&laquo;</button>
            <button id="nav-prev" class="slide-nav-btn" title="Previous slide" aria-label="Previous slide">&lsaquo;</button>
        </div>
        <div class="slide-nav-group slide-nav-right">
            <button id="nav-next" class="slide-nav-btn" title="Next slide" aria-label="Next slide">&rsaquo;</button>
            <button id="nav-fast-next" class="slide-nav-btn" title="Forward 5 slides" aria-label="Forward 5 slides">&raquo;</button>
        </div>
    </div>

    <!-- Reveal.js -->
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/markdown/markdown.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/highlight/highlight.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/math/math.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/notes/notes.js"></script>

    <script>
        Reveal.initialize({{
            hash: true,
            slideNumber: 'c/t',
            showSlideNumber: 'all',
            transition: 'none',
            transitionSpeed: 'fast',
            backgroundTransition: 'none',
            center: true,
            width: 1680,
            height: 945,
            margin: 0.03,
            minScale: 0.5,
            maxScale: 1.2,
            controls: false,
            plugins: [
                RevealHighlight,
                RevealMath.KaTeX,
                RevealNotes
            ]
        }});

        /* Custom slide navigation */
        function navTo(delta) {{
            const total = Reveal.getTotalSlides();
            const cur = Reveal.getState().indexh;
            Reveal.slide(Math.max(0, Math.min(total - 1, cur + delta)));
        }}
        document.getElementById('nav-fast-prev').addEventListener('click', () => navTo(-5));
        document.getElementById('nav-prev').addEventListener('click', () => navTo(-1));
        document.getElementById('nav-next').addEventListener('click', () => navTo(1));
        document.getElementById('nav-fast-next').addEventListener('click', () => navTo(5));

        /* Overflow debug overlay — activate with ?debug in URL */
        if (new URLSearchParams(window.location.search).has('debug')) {{
            Reveal.on('ready', () => {{
                const max = 865; /* 945px slide height - 80px vertical padding */
                document.querySelectorAll('.reveal .slides > section').forEach((s, i) => {{
                    const h = s.scrollHeight;
                    const label = document.createElement('div');
                    label.style.cssText = 'position:absolute;top:4px;right:4px;font-size:12px;z-index:999;padding:2px 6px;border-radius:4px;font-family:monospace;';
                    label.style.color = '#000';
                    if (h > max) {{
                        s.style.outline = '2px solid #f87171';
                        label.style.background = '#f87171';
                        label.textContent = h + 'px (+' + (h - max) + ' overflow)';
                    }} else {{
                        s.style.outline = '2px solid #34d399';
                        label.style.background = '#34d399';
                        label.textContent = h + 'px (' + (max - h) + 'px free)';
                    }}
                    s.style.position = 'relative';
                    s.appendChild(label);
                }});
            }});
        }}
    </script>

    <!-- Custom charts -->
    <script src="/static/js/charts.js"></script>

    <!-- Diffusion transition engine -->
    <script src="/static/js/diffusion.js"></script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def presentation():
    slides = load_slides()
    return build_presentation_html(slides)


@app.get("/reload", response_class=HTMLResponse)
async def reload():
    """Force reload slides (useful during editing)."""
    slides = load_slides()
    return build_presentation_html(slides)


if __name__ == "__main__":
    import uvicorn

    print("Starting RLVR presentation server...")
    print("Open http://localhost:8000 in your browser")
    print("Press 's' in the presentation for speaker notes")
    print("Press 'o' for slide overview")
    print("Press 'f' for fullscreen")
    uvicorn.run(app, host="0.0.0.0", port=8000)
