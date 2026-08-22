#!/usr/bin/env python3
"""Generate the five-seed parent-fingerprint probe figure as SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

LABELS = {
    "rgb": "RGB",
    "grayscale": "Grayscale",
    "stain_normalized": "Stain-normalized",
    "background_only": "Background only",
    "highpass": "High-pass",
    "shuffled": "Shuffled labels",
}


def figure_svg(summary: pd.DataFrame, control: dict[str, object]) -> str:
    required = {"image_mode", "shuffled_labels", "n", "mean", "sd", "complete"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Probe summary misses columns: {missing}")
    if not summary.complete.astype(bool).all() or (summary.n < 5).any():
        raise ValueError("Figure 2 requires complete five-seed probe cells")
    observed = summary.loc[~summary.shuffled_labels.astype(bool)].set_index("image_mode")
    shuffled = summary.loc[summary.shuffled_labels.astype(bool)]
    modes = ["rgb", "grayscale", "stain_normalized", "background_only", "highpass"]
    if any(mode not in observed.index for mode in modes) or len(shuffled) != 1:
        raise ValueError("Figure 2 requires five observed modes and one shuffled RGB control")
    values = [(mode, float(observed.loc[mode, "mean"]), float(observed.loc[mode, "sd"])) for mode in modes]
    values.append(("shuffled", float(shuffled.iloc[0]["mean"]), float(shuffled.iloc[0]["sd"])))
    chance = float(control["chance"])
    scale_max = max(0.30, max(mean + sd for _, mean, sd in values) * 1.12)
    plot_bottom, plot_height = 535, 390

    elements = []
    for index, (mode, mean, sd) in enumerate(values):
        x = 115 + index * 170
        height = mean / scale_max * plot_height
        error = sd / scale_max * plot_height
        y = plot_bottom - height
        color = "#C43C4E" if mode == "rgb" else ("#84919D" if mode == "shuffled" else "#2A6F97")
        elements.append(
            f'<rect x="{x}" y="{y:.1f}" width="105" height="{height:.1f}" rx="7" fill="{color}"/>'
            f'<line x1="{x + 52.5}" y1="{y - error:.1f}" x2="{x + 52.5}" y2="{y + error:.1f}" stroke="#17212B" stroke-width="3"/>'
            f'<line x1="{x + 35}" y1="{y - error:.1f}" x2="{x + 70}" y2="{y - error:.1f}" stroke="#17212B" stroke-width="3"/>'
            f'<text x="{x + 52.5}" y="{max(142, y - error - 13):.1f}" text-anchor="middle" class="number">{100 * mean:.2f} ± {100 * sd:.2f}%</text>'
            f'<text x="{x + 52.5}" y="575" text-anchor="middle" class="label">{escape(LABELS[mode])}</text>'
        )
    chance_y = plot_bottom - chance / scale_max * plot_height
    multiplier = values[0][1] / chance
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680">
<style>
  .title {{ font: 700 31px Arial, sans-serif; fill: #17212B; }}
  .subtitle {{ font: 19px Arial, sans-serif; fill: #435466; }}
  .label {{ font: 16px Arial, sans-serif; fill: #17212B; }}
  .number {{ font: 700 15px Arial, sans-serif; fill: #17212B; }}
  .axis {{ font: 15px Arial, sans-serif; fill: #435466; }}
</style>
<rect width="1200" height="680" fill="#FAFBFC"/>
<text x="60" y="55" class="title">Cell crops carry a measurable parent-field fingerprint</text>
<text x="60" y="88" class="subtitle">ResNet-18 parent identification; balanced 424-way task; mean ± sample SD over five fixed seeds.</text>
<line x1="80" y1="535" x2="1130" y2="535" stroke="#667788" stroke-width="2"/>
<line x1="80" y1="145" x2="80" y2="535" stroke="#667788" stroke-width="2"/>
<text x="27" y="345" transform="rotate(-90 27 345)" class="axis">Parent-ID test accuracy</text>
<line x1="80" y1="{chance_y:.1f}" x2="1130" y2="{chance_y:.1f}" stroke="#8C6D1F" stroke-width="3" stroke-dasharray="9 7"/>
<text x="1130" y="{chance_y - 9:.1f}" text-anchor="end" class="number">chance = {100 * chance:.4f}%</text>
{''.join(elements)}
<rect x="72" y="620" width="1056" height="42" rx="9" fill="#EAF1F5"/>
<text x="600" y="647" text-anchor="middle" class="number">RGB is {multiplier:.1f}× chance; shuffled-label 95% CI covers chance: {bool(control['passed'])}.</text>
</svg>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("control")
    parser.add_argument("output")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    control = json.loads(Path(args.control).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(figure_svg(summary, control))
    print(output)


if __name__ == "__main__":
    main()
