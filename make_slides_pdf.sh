#!/usr/bin/env bash
# Render the reveal.js presentation to PDF (one slide per page).
#
# Requires the FastAPI server to already be serving on :8000.
# Start it separately: `python3 app.py`
#
# Output: ./presentation.pdf

set -euo pipefail

URL="http://localhost:8000/?print-pdf"
OUT="$(dirname "$0")/presentation.pdf"

if ! curl -fsS -o /dev/null --max-time 2 "http://localhost:8000/"; then
  echo "Error: nothing serving at http://localhost:8000/" >&2
  echo "Start the presentation server first:" >&2
  echo "    python3 app.py" >&2
  exit 1
fi

CHROME="${CHROME:-google-chrome}"
if ! command -v "$CHROME" >/dev/null 2>&1; then
  echo "Error: $CHROME not found on PATH. Set CHROME=<path> to override." >&2
  exit 1
fi

echo "Rendering $URL → $OUT"
"$CHROME" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --hide-scrollbars \
  --no-pdf-header-footer \
  --virtual-time-budget=30000 \
  --print-to-pdf="$OUT" \
  "$URL"

echo "Done: $OUT"
