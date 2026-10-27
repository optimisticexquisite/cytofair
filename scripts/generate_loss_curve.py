"""Render a frozen five-seed ResNet-50 P1/P2 training-loss trajectory as SVG."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COLORS = {"P1": "#C43C4E", "P2": "#2A6F97"}


def _load_histories(results_dir: Path, protocol: str) -> list[list[float]]:
    paths = sorted(results_dir.glob(f"sipakmed_resnet50_{protocol.lower()}_seed*.json"))
    histories: list[list[float]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != protocol or payload.get("architecture") != "resnet50":
            continue
        history = [float(row["train_loss"]) for row in payload["history"]]
        if len(history) != 20:
            raise ValueError(f"Expected 20 epochs in {path}, found {len(history)}")
        histories.append(history)
    if len(histories) != 5:
        raise ValueError(f"Expected five {protocol} histories, found {len(histories)}")
    return histories


def _series(histories: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [statistics.mean(values) for values in zip(*histories)]
    sds = [statistics.stdev(values) for values in zip(*histories)]
    return means, sds


def figure_svg(results_dir: Path) -> str:
    series = {protocol: _series(_load_histories(results_dir, protocol)) for protocol in COLORS}
    left, right, top, bottom = 105, 940, 105, 440
    width, height = right - left, bottom - top
    maximum = max(mean + sd for means, sds in series.values() for mean, sd in zip(means, sds))
    maximum = max(0.75, maximum * 1.05)

    grid: list[str] = []
    for index in range(5):
        value = maximum * index / 4
        y = bottom - value / maximum * height
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#D7DEE4" stroke-width="1"/>'
            f'<text x="{left - 13}" y="{y + 5:.1f}" text-anchor="end" class="axis">{value:.2f}</text>'
        )

    curves: list[str] = []
    for protocol, (means, sds) in series.items():
        upper: list[str] = []
        lower: list[str] = []
        line: list[str] = []
        for index, (mean, sd) in enumerate(zip(means, sds)):
            x = left + index / 19 * width
            y = bottom - mean / maximum * height
            y_upper = bottom - min(maximum, mean + sd) / maximum * height
            y_lower = bottom - max(0.0, mean - sd) / maximum * height
            line.append(f"{x:.1f},{y:.1f}")
            upper.append(f"{x:.1f},{y_upper:.1f}")
            lower.append(f"{x:.1f},{y_lower:.1f}")
        band = " ".join(upper + list(reversed(lower)))
        curves.append(
            f'<polygon points="{band}" fill="{COLORS[protocol]}" opacity="0.14"/>'
            f'<polyline points="{" ".join(line)}" fill="none" stroke="{COLORS[protocol]}" stroke-width="4"/>'
        )

    ticks: list[str] = []
    for epoch in (1, 5, 10, 15, 20):
        x = left + (epoch - 1) / 19 * width
        ticks.append(f'<text x="{x:.1f}" y="472" text-anchor="middle" class="axis">{epoch}</text>')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="540" viewBox="0 0 1000 540">
<style>
  .title {{ font: 700 27px Arial, sans-serif; fill: #17212B; }}
  .subtitle {{ font: 17px Arial, sans-serif; fill: #435466; }}
  .axis {{ font: 14px Arial, sans-serif; fill: #435466; }}
  .legend {{ font: 16px Arial, sans-serif; fill: #17212B; }}
</style>
<rect width="1000" height="540" fill="#FAFBFC"/>
<text x="50" y="42" class="title">Training loss converges under both split protocols</text>
<text x="50" y="70" class="subtitle">ResNet-50 on SIPaKMeD; mean line and ±1 sample-SD band over five fixed seeds.</text>
{''.join(grid)}
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#667788" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#667788" stroke-width="2"/>
{''.join(curves)}
{''.join(ticks)}
<text x="520" y="505" text-anchor="middle" class="axis">Epoch</text>
<text x="28" y="285" transform="rotate(-90 28 285)" class="axis">Training cross-entropy</text>
<line x1="700" y1="96" x2="745" y2="96" stroke="{COLORS['P1']}" stroke-width="4"/>
<text x="755" y="102" class="legend">P1 crop split</text>
<line x1="700" y1="124" x2="745" y2="124" stroke="{COLORS['P2']}" stroke-width="4"/>
<text x="755" y="130" class="legend">P2 parent-disjoint</text>
</svg>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(figure_svg(args.results_dir))
    print(args.output)


if __name__ == "__main__":
    main()
