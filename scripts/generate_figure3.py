#!/usr/bin/env python3
"""Generate the matched P1/P1-prime/P2/P3 classifier comparison as SVG."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

PROTOCOLS = ("P1", "P1-prime", "P2", "P3")
COLORS = {
    "P1": "#C43C4E",
    "P1-prime": "#D8894C",
    "P2": "#2A6F97",
    "P3": "#315B55",
}
ARCHITECTURE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v2": "MobileNet-V2",
    "vit_b_16": "ViT-B/16",
}


def figure_svg(summary: pd.DataFrame, *, dataset: str = "SIPaKMeD") -> str:
    required = {"dataset", "architecture", "protocol", "metric", "n", "mean", "sd", "complete"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Classification summary misses columns: {missing}")
    selected = summary.loc[
        (summary.dataset == dataset) & (summary.metric == "accuracy")
    ].copy()
    if selected.empty:
        raise ValueError(f"No accuracy rows found for {dataset}")
    if not selected.complete.astype(bool).all() or (selected.n < 5).any():
        raise ValueError("Figure 3 requires complete five-seed classification cells")
    if selected.duplicated(["architecture", "protocol"]).any():
        raise ValueError("Figure 3 received duplicate architecture/protocol cells")
    architectures = [
        architecture
        for architecture in ARCHITECTURE_LABELS
        if architecture in set(selected.architecture)
    ]
    if not architectures:
        raise ValueError("Figure 3 contains no recognised architecture")
    indexed = selected.set_index(["architecture", "protocol"])
    expected = {(architecture, protocol) for architecture in architectures for protocol in PROTOCOLS}
    missing_cells = sorted(expected - set(indexed.index))
    if missing_cells:
        raise ValueError(f"Figure 3 misses cells: {missing_cells}")

    minimum = max(0.0, float(selected["mean"].min() - selected["sd"].max() - 0.05))
    maximum = min(1.0, float(selected["mean"].max() + selected["sd"].max() + 0.04))
    if maximum - minimum < 0.12:
        midpoint = (maximum + minimum) / 2
        minimum, maximum = max(0.0, midpoint - 0.06), min(1.0, midpoint + 0.06)
    plot_left, plot_right, plot_top, plot_bottom = 105, 1170, 155, 595
    plot_height = plot_bottom - plot_top
    group_width = (plot_right - plot_left) / len(architectures)
    bar_width, bar_gap = min(31, group_width / 6), 7
    elements: list[str] = []

    for tick_index in range(6):
        value = minimum + tick_index * (maximum - minimum) / 5
        y = plot_bottom - (value - minimum) / (maximum - minimum) * plot_height
        elements.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            'stroke="#D7DEE4" stroke-width="1"/>'
            f'<text x="{plot_left - 14}" y="{y + 5:.1f}" text-anchor="end" class="axis">'
            f'{100 * value:.0f}%</text>'
        )

    for architecture_index, architecture in enumerate(architectures):
        group_center = plot_left + group_width * (architecture_index + 0.5)
        total_width = len(PROTOCOLS) * bar_width + (len(PROTOCOLS) - 1) * bar_gap
        first_x = group_center - total_width / 2
        for protocol_index, protocol in enumerate(PROTOCOLS):
            row = indexed.loc[(architecture, protocol)]
            mean, sd = float(row["mean"]), float(row["sd"])
            x = first_x + protocol_index * (bar_width + bar_gap)
            y = plot_bottom - (mean - minimum) / (maximum - minimum) * plot_height
            height = plot_bottom - y
            error = sd / (maximum - minimum) * plot_height
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" '
                f'rx="4" fill="{COLORS[protocol]}"/>'
                f'<line x1="{x + bar_width / 2:.1f}" y1="{y - error:.1f}" '
                f'x2="{x + bar_width / 2:.1f}" y2="{y + error:.1f}" '
                'stroke="#17212B" stroke-width="2"/>'
                f'<line x1="{x + 4:.1f}" y1="{y - error:.1f}" '
                f'x2="{x + bar_width - 4:.1f}" y2="{y - error:.1f}" '
                'stroke="#17212B" stroke-width="2"/>'
            )
        elements.append(
            f'<text x="{group_center:.1f}" y="630" text-anchor="middle" class="label">'
            f'{escape(ARCHITECTURE_LABELS[architecture])}</text>'
        )

    legend = []
    for index, protocol in enumerate(PROTOCOLS):
        x = 300 + index * 180
        display = "P1′" if protocol == "P1-prime" else protocol
        legend.append(
            f'<rect x="{x}" y="102" width="24" height="18" rx="3" fill="{COLORS[protocol]}"/>'
            f'<text x="{x + 34}" y="117" class="legend">{display}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="690" viewBox="0 0 1250 690">
<style>
  .title {{ font: 700 30px Arial, sans-serif; fill: #17212B; }}
  .subtitle {{ font: 18px Arial, sans-serif; fill: #435466; }}
  .label {{ font: 15px Arial, sans-serif; fill: #17212B; }}
  .legend {{ font: 16px Arial, sans-serif; fill: #17212B; }}
  .axis {{ font: 14px Arial, sans-serif; fill: #435466; }}
</style>
<rect width="1250" height="690" fill="#FAFBFC"/>
<text x="55" y="50" class="title">Parent-disjoint evaluation changes measured accuracy</text>
<text x="55" y="80" class="subtitle">{escape(dataset)} test accuracy; mean ± sample SD over five fixed seeds. P1′ is training-size matched to P2.</text>
{''.join(legend)}
<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#667788" stroke-width="2"/>
<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#667788" stroke-width="2"/>
<text x="35" y="390" transform="rotate(-90 35 390)" class="axis">Test accuracy</text>
{''.join(elements)}
</svg>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("output")
    parser.add_argument("--dataset", default="SIPaKMeD")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(figure_svg(pd.read_csv(args.summary), dataset=args.dataset))
    print(output)


if __name__ == "__main__":
    main()
