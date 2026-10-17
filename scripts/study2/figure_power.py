#!/usr/bin/env python3
"""Supplementary figure: simulated power of the registered H1 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import figstyle as style
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power", default="results/study2/power/power_power.csv")
    parser.add_argument("--output", default="paper/v2/figures/power.pdf")
    args = parser.parse_args()
    style.apply()
    power = pd.read_csv(args.power)
    power = power.loc[power.delta == 0.8]
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    panels = ((1.0, "Pilot variances", 0.90, (20.6, 0.50)), (1.5, "All variances inflated 1.5x", 0.75, (20.6, 0.38)))
    for ax, (scale, title, threshold, label_at) in zip(axes, panels, strict=True):
        subset = power.loc[power.scale == scale]
        for color, (_, group) in zip(style.ORDINAL_BLUE, subset.groupby("architectures"), strict=True):
            group = group.sort_values("seeds")
            ax.plot(group.seeds, group.power, color=color, zorder=3)
            style.ring(ax, group.seeds, group.power, color, size=18)
        ax.axhline(threshold, color=style.MUTED, linewidth=0.8, zorder=2)
        ax.annotate(f"{threshold:.2f}", (1.0, threshold), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7, color=style.INK_2)
        chosen = subset.loc[(subset.seeds == 20) & (subset.architectures == 12)].iloc[0]
        style.ring(ax, [20], [chosen.power], style.ORANGE, size=46, zorder=5)
        ax.annotate(
            f"registered design\n20 seeds, 12 models\npower {chosen.power:.3f}",
            (20, chosen.power), xytext=label_at, textcoords="data", fontsize=7, color=style.INK, ha="left",
            arrowprops={"arrowstyle": "-", "color": style.MUTED, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 4},
        )
        ax.set_title(title, loc="left", color=style.INK)
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.set_xlim(9, 31)
        ax.set_xlabel("Seeds (splits per protocol)")
    axes[0].set_ylabel("Power for a 0.8-pp effect")
    axes[0].set_ylim(0.3, 1.02)
    handles = [Line2D([], [], color=color, marker="o", markersize=4, markeredgecolor="white", linewidth=1.5)
               for color in style.ORDINAL_BLUE]
    figure.legend(handles, ["6", "8", "10", "12"], title="Architectures", loc="center right",
                  bbox_to_anchor=(1.07, 0.5), title_fontsize=7.5, labelcolor=style.INK_2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))
    print(output)


if __name__ == "__main__":
    main()
