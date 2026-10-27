#!/usr/bin/env bash
# Build the ISBI 2027 paper from the committed results.
#   1. copy the generated number macros (paper/v2 and record scripts),
#   2. derive the paper-only numbers and redraw the data figures,
#   3. compile with tectonic (XeLaTeX + BibTeX) and keep the log for checking.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
mkdir -p "$here/generated"
cp "$repo/paper/v2/generated/numbers.tex" "$here/generated/numbers.tex"
cp "$repo/record/generated/record_numbers.tex" "$here/generated/record_numbers.tex"
(cd "$repo" && python3 paper/isbi2027/derived_numbers.py >/dev/null && python3 paper/isbi2027/make_figures.py >/dev/null 2>&1)
cd "$here"
tectonic -X compile --keep-logs --keep-intermediates main.tex
echo "built $here/main.pdf"
