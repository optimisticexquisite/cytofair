"""Shared matplotlib style for Study 2 figures (static, print-first).

Palette: the validated reference instance (categorical slots in fixed order,
an ordinal blue ramp, recessive chrome). Text always uses ink tokens, never a
series colour; grids and baselines are solid hairlines.
"""

from __future__ import annotations

import matplotlib as mpl

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
GRAY_SERIES = "#a9a8a2"
ORDINAL_BLUE = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
CLEAN, LEAK = BLUE, ORANGE  # identity is fixed across every figure


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "regular",
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def zero_line(ax, orientation: str = "vertical") -> None:
    """Solid hairline reference at zero (never dashed)."""
    if orientation == "vertical":
        ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    else:
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)


def ring(ax, x, y, color, size=26, marker="o", zorder=4, **kwargs):
    """Filled marker with a white surface ring so it stays legible over lines."""
    return ax.scatter(x, y, s=size, color=color, marker=marker, edgecolors="white",
                      linewidths=0.9, zorder=zorder, **kwargs)
