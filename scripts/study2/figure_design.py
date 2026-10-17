#!/usr/bin/env python3
"""Schematic of the 2C leakage-by-diversity factorial (one class, not to scale)."""

from __future__ import annotations

import argparse
from pathlib import Path

import figstyle as style
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

N_EVAL, N_POOL, N_LOW, N_UNSEEN = 6, 20, 14, 3
CELL = 0.82


def square(ax, x, y, face, *, hatch=None, edge="white", alpha=1.0):
    ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=face, edgecolor=edge, linewidth=0.8,
                           hatch=hatch, alpha=alpha))


def split_square(ax, x, y, donor_face):
    """Evaluation field: lower half test crops (ink), upper half donor siblings."""
    ax.add_patch(Rectangle((x, y), CELL, CELL / 2, facecolor=style.INK_2, edgecolor="white", linewidth=0.8))
    ax.add_patch(Rectangle((x, y + CELL / 2), CELL, CELL / 2, facecolor=donor_face, edgecolor="white", linewidth=0.8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/v2/figures/design.pdf")
    args = parser.parse_args()
    style.apply()
    figure, ax = plt.subplots(figsize=(6.6, 2.05))
    ax.set_axis_off()
    pale = "#f0efec"
    rows = [
        ("Fields", None),
        ("leak, high diversity", ("leak", N_POOL)),
        ("clean, high diversity", ("clean", N_POOL)),
        ("leak, low diversity", ("leak", N_LOW)),
        ("clean, low diversity", ("clean", N_LOW)),
    ]
    x_eval, x_pool = 0.0, N_EVAL + 0.8
    x_unseen = x_pool + N_POOL + 0.8
    for row_index, (label, spec) in enumerate(rows):
        y = -row_index * 1.15
        ax.text(-0.4, y + CELL / 2, label, ha="right", va="center", fontsize=8,
                color=style.INK if spec is None else style.INK_2)
        if spec is None:
            for i in range(N_EVAL):
                split_square(ax, x_eval + i, y, style.LEAK)
            for i in range(N_POOL):
                square(ax, x_pool + i, y, style.CLEAN)
            for i in range(N_UNSEEN):
                square(ax, x_unseen + i, y, style.GRAY_SERIES)
            continue
        kind, used = spec
        for i in range(N_EVAL):
            if kind == "leak":
                split_square(ax, x_eval + i, y, style.LEAK)
            else:
                ax.add_patch(Rectangle((x_eval + i, y), CELL, CELL / 2, facecolor=style.INK_2,
                                       edgecolor="white", linewidth=0.8))
                ax.add_patch(Rectangle((x_eval + i, y + CELL / 2), CELL, CELL / 2, facecolor=pale,
                                       edgecolor="white", linewidth=0.8))
        n_pool = used - N_EVAL if kind == "leak" else used
        for i in range(N_POOL):
            square(ax, x_pool + i, y, style.CLEAN if i < n_pool else pale)
        for i in range(N_UNSEEN):
            square(ax, x_unseen + i, y, pale)
        ax.text(x_unseen + N_UNSEEN + 0.5, y + CELL / 2, f"{used} fields", ha="left", va="center",
                fontsize=7.5, color=style.INK_2)
    top = CELL + 0.25
    ax.text(x_eval + N_EVAL / 2, top, "evaluation fields E", ha="center", va="bottom", fontsize=7.5, color=style.INK)
    ax.text(x_pool + N_POOL / 2, top, "clean pool B (one fixed permutation)", ha="center", va="bottom",
            fontsize=7.5, color=style.INK)
    ax.text(x_unseen + N_UNSEEN / 2, top, "unseen U", ha="center", va="bottom", fontsize=7.5, color=style.INK)
    from matplotlib.patches import Patch

    entries = [
        (style.INK_2, "common test crops (every cell)"),
        (style.LEAK, "donor siblings of test crops"),
        (style.CLEAN, "clean-pool crops"),
        (style.GRAY_SERIES, "transplant backgrounds only"),
    ]
    handles = [Patch(facecolor=face, edgecolor="white", label=text) for face, text in entries]
    ax.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0),
                  handlelength=1.0, handleheight=1.0, columnspacing=1.6, fontsize=7.5)
    bottom = -(len(rows) - 1) * 1.15 - 0.25
    ax.set_xlim(-7.5, x_unseen + N_UNSEEN + 4.5)
    ax.set_ylim(bottom, top + 0.8)
    ax.set_aspect("equal")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))
    print(output)


if __name__ == "__main__":
    main()
