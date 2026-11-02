#!/usr/bin/env python3
"""Generate the paper's self-contained thesis figure as publication-ready SVG."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


def _leading_number(value: str) -> float:
    match = re.match(r"\s*([0-9.]+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse mean from {value!r}")
    return float(match.group(1))


def figure_svg(table: pd.DataFrame) -> str:
    values = {
        row.dataset: _leading_number(row.observed_p1_straddle_pct_mean_sd)
        for row in table.itertuples(index=False)
    }
    order = [name for name in ("SIPaKMeD", "Cx22", "CRIC", "Herlev") if name in values]
    bars = []
    for index, name in enumerate(order):
        y = 500 + 54 * index
        width = 4.1 * values[name]
        color = "#C43C4E" if values[name] > 0 else "#2A6F97"
        bars.append(
            f'<text x="760" y="{y + 20}" class="label">{escape(name)}</text>'
            f'<rect x="885" y="{y}" width="{width:.1f}" height="28" rx="4" fill="{color}"/>'
            f'<text x="{895 + width:.1f}" y="{y + 20}" class="number">{values[name]:.2f}%</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="780" viewBox="0 0 1400 780">
<style>
  .title {{ font: 700 32px Arial, sans-serif; fill: #17212B; }}
  .subtitle {{ font: 20px Arial, sans-serif; fill: #435466; }}
  .head {{ font: 700 23px Arial, sans-serif; fill: #17212B; }}
  .label {{ font: 18px Arial, sans-serif; fill: #17212B; }}
  .small {{ font: 16px Arial, sans-serif; fill: #435466; }}
  .number {{ font: 700 17px Arial, sans-serif; fill: #17212B; }}
</style>
<rect width="1400" height="780" fill="#FAFBFC"/>
<text x="60" y="58" class="title">Crop-level splitting leaks parent identity</text>
<text x="60" y="92" class="subtitle">Sibling cells inherit stain, illumination, focus, background—and sometimes literal pixels—from one field.</text>

<text x="70" y="155" class="head">1. One parent field yields several crops</text>
<rect x="75" y="185" width="390" height="270" rx="14" fill="#E9D7CF" stroke="#7D5A50" stroke-width="3"/>
<circle cx="165" cy="275" r="46" fill="#B07A8F" opacity=".75"/><circle cx="330" cy="250" r="55" fill="#916B9C" opacity=".65"/>
<circle cx="245" cy="380" r="52" fill="#C1848A" opacity=".7"/><circle cx="390" cy="370" r="42" fill="#8B719E" opacity=".65"/>
<rect x="108" y="218" width="112" height="112" fill="none" stroke="#2A6F97" stroke-width="5"/>
<rect x="274" y="195" width="120" height="120" fill="none" stroke="#2A6F97" stroke-width="5"/>
<rect x="188" y="326" width="116" height="112" fill="none" stroke="#2A6F97" stroke-width="5"/>
<rect x="340" y="320" width="104" height="105" fill="none" stroke="#2A6F97" stroke-width="5"/>
<text x="160" y="482" class="small">parent_id = P17</text>

<path d="M475 305 L560 305" stroke="#667788" stroke-width="4" marker-end="url(#arrow)"/>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#667788"/></marker></defs>

<text x="550" y="155" class="head">2. Random crop split scatters siblings</text>
<rect x="560" y="190" width="175" height="120" rx="10" fill="#D9EAF3" stroke="#2A6F97" stroke-width="3"/>
<rect x="760" y="190" width="175" height="120" rx="10" fill="#F6D9DE" stroke="#C43C4E" stroke-width="3"/>
<rect x="560" y="335" width="175" height="120" rx="10" fill="#D9EAF3" stroke="#2A6F97" stroke-width="3"/>
<rect x="760" y="335" width="175" height="120" rx="10" fill="#F6D9DE" stroke="#C43C4E" stroke-width="3"/>
<text x="615" y="250" class="head">TRAIN</text><text x="810" y="250" class="head">TEST</text>
<text x="615" y="395" class="head">TRAIN</text><text x="810" y="395" class="head">TEST</text>
<path d="M610 175 C680 115, 810 115, 885 175" fill="none" stroke="#C43C4E" stroke-width="5" stroke-dasharray="10 8"/>
<text x="643" y="124" class="number">same parent on both sides → leakage</text>

<text x="1010" y="155" class="head">3. Parent-disjoint split</text>
<rect x="1045" y="190" width="260" height="265" rx="14" fill="#D9EAF3" stroke="#2A6F97" stroke-width="3"/>
<text x="1126" y="230" class="head">TRAIN</text>
<rect x="1080" y="260" width="85" height="70" rx="7" fill="#A8CEE2"/><rect x="1185" y="260" width="85" height="70" rx="7" fill="#A8CEE2"/>
<rect x="1080" y="350" width="85" height="70" rx="7" fill="#A8CEE2"/><rect x="1185" y="350" width="85" height="70" rx="7" fill="#A8CEE2"/>
<text x="1070" y="482" class="small">all P17 crops stay together</text>

<line x1="60" y1="505" x2="700" y2="505" stroke="#D7DEE5" stroke-width="2"/>
<text x="70" y="552" class="head">Observed parent straddling under P1 (mean of 5 seeds)</text>
<text x="70" y="590" class="label">The effect follows construction:</text>
<text x="70" y="625" class="small">• cropped-from-parent datasets straddle</text>
<text x="70" y="658" class="small">• Herlev (one cell per parent) is exactly zero</text>
<text x="70" y="705" class="number">Finding: crop-level evaluation is non-independent; group by parent before splitting.</text>

<line x1="760" y1="505" x2="1325" y2="505" stroke="#D7DEE5" stroke-width="2"/>
{''.join(bars)}
<text x="885" y="735" class="small">0</text><text x="1085" y="735" class="small">50%</text><text x="1280" y="735" class="small">100%</text>
<line x1="885" y1="720" x2="1295" y2="720" stroke="#667788" stroke-width="2"/>
</svg>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table1")
    parser.add_argument("output")
    args = parser.parse_args()
    table = pd.read_csv(args.table1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(figure_svg(table))
    print(output)


if __name__ == "__main__":
    main()
