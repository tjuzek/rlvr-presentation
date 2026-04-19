#!/usr/bin/env bash
# Render the unified RLVR demo report to PDF (Overview + Run 1..4 in order).
#
# Requires a local static server on :8765 rooted at this demo/ directory.
# Start it separately:
#     cd demo && python3 -m http.server 8765
#
# Output: ./results/rlvr_demo_report.pdf

set -euo pipefail

URL="http://127.0.0.1:8765/results/rlvr_demo_report.html?print=1"
OUT="$(dirname "$0")/results/rlvr_demo_report.pdf"

if ! curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:8765/results/rlvr_demo_report.html"; then
  echo "Error: nothing serving rlvr_demo_report.html on :8765" >&2
  echo "Start the static server first (from this demo/ directory):" >&2
  echo "    python3 -m http.server 8765" >&2
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
  --virtual-time-budget=20000 \
  --print-to-pdf="$OUT" \
  "$URL"

echo "Done: $OUT"
