#!/usr/bin/env python3
"""Amendment 8 follow-up figures, drawn only from the registered follow-up outputs.

* ``carriers.pdf``: the leak effect under each test-time ablation (F2), under each
  training condition (F3, F4), and the CRIC slide-context analysis (F1).
* ``dose.pdf``: P1 - P2 gap against parent recoverability along each fingerprint
  manipulation (LEACE ranks, F5; stain normalisation, F6; linear adversary, F7).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import figstyle as style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ABLATION_NAMES = {
    "original": "original", "grey": "greyscale", "stain_transfer": "stain transfer", "low_pass": "low-pass",
    "illumination_flat": "flat illumination", "background_grey": "background removed",
}
CONDITION_NAMES = {"standard": "standard", "masked": "background masked", "colour": "colour augmented"}


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def t_interval(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if len(values) < 2:
        return mean, mean, mean
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def fmt(record, key="estimate") -> str:
    if not record or record.get(key) is None:
        return "pending"
    text = f"{record['estimate']:.2f} [{record['ci95'][0]:.2f}, {record['ci95'][1]:.2f}]"
    return text.replace("-", "−")


def leak_effects(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Leak minus clean accuracy (pp) per architecture, seed, diversity level, and ``group``."""
    frame = frame.assign(div=frame.cell.str.split("_").str[1], leak=frame.cell.str.startswith("leak"))
    wide = frame.pivot_table(index=["architecture", "seed", "div", group], columns="leak", values="accuracy")
    return (100 * (wide[True] - wide[False])).rename("effect").reset_index()


def rows_panel(ax, rows, *, color, xlabel, highlight=None):
    for position, (label, values) in enumerate(rows):
        mean, low, high = t_interval(values)
        shade = color if highlight is None or label == highlight else style.GRAY_SERIES
        ax.plot([low, high], [position, position], color=shade, linewidth=1.3, zorder=2)
        style.ring(ax, [mean], [position], shade, size=24)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    style.zero_line(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def figure_carriers(root: Path, output: Path) -> None:
    tables = root / "tables"
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.15, 1.05, 1.0]})

    ablations = pd.read_csv(tables / "ablation_runs.csv")
    effects = leak_effects(ablations, "ablation")
    rows = [(ABLATION_NAMES[name], effects.loc[effects.ablation == name, "effect"]) for name in ABLATION_NAMES]
    rows_panel(axes[0], rows, color=style.LEAK, xlabel="Leak minus clean (pp)", highlight="stain transfer")
    axes[0].set_title("Test-time ablations (F2)", loc="left", color=style.INK)

    carrier = pd.read_csv(tables / "carrier_runs.csv")
    effects = leak_effects(carrier, "condition")
    rows = [(CONDITION_NAMES[name], effects.loc[effects.condition == name, "effect"]) for name in CONDITION_NAMES]
    rows_panel(axes[1], rows, color=style.LEAK, xlabel="Leak minus clean (pp)")
    axes[1].set_title("Carrier removed in training (F3, F4)", loc="left", color=style.INK)

    cric = load(root / "cric_context.json")
    ax = axes[2]
    if cric is not None:
        items = [
            ("accuracy, concordant", cric["accuracy_difference_concordant"]),
            ("accuracy, discordant", cric["accuracy_difference_discordant"]),
            ("context-class prob.", cric["F1"]),
        ]
        for position, (label, record) in enumerate(items):
            color = style.BLUE if position < 2 else style.INK_2
            ax.plot(record["ci95"], [position, position], color=color, linewidth=1.3, zorder=2)
            style.ring(ax, [record["estimate"]], [position], color, size=24)
        ax.set_yticks(range(len(items)), [label for label, _ in items])
        ax.invert_yaxis()
    style.zero_line(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("P1 minus P2 (pp)")
    ax.set_title("CRIC slide context (F1)", loc="left", color=style.INK)
    figure.tight_layout(w_pad=1.2)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def pairs(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    frame = frame.assign(value=100 * frame.accuracy, R=100 * frame.within_class_accuracy)
    wide = frame.pivot_table(index=["architecture", "seed", level], columns="protocol", values=["value", "R"])
    out = pd.DataFrame({"gap": wide[("value", "P1")] - wide[("value", "P2")],
                        "R_pair": (wide[("R", "P1")] + wide[("R", "P2")]) / 2}).reset_index()
    return out


def dose_panel(ax, frame, level, label, color, checks, title):
    included = [arch for arch in frame.architecture.unique() if checks.get(arch, {}).get("manipulation_check_passed")]
    if not checks:
        included = list(frame.architecture.unique())
    for arch, group in frame.groupby("architecture"):
        means = group.groupby(level)[["R_pair", "gap"]].mean()
        ax.plot(means.R_pair, means.gap, color=style.GRAY_SERIES if arch in included else style.GRID,
                linewidth=0.8, zorder=1)
    pooled = frame.loc[frame.architecture.isin(included)].groupby(level)[["R_pair", "gap"]].mean()
    if len(pooled):
        ax.plot(pooled.R_pair, pooled.gap, color=color, linewidth=1.6, zorder=2)
        style.ring(ax, pooled.R_pair, pooled.gap, color, size=26, zorder=3)
        for value, row in pooled.iterrows():
            ax.annotate(label(value), (row.R_pair, row.gap), xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color=style.INK_2)
    style.zero_line(ax, "horizontal")
    ax.set_xlabel("Recoverability R (%)")
    ax.set_title(title, loc="left", color=style.INK)


def figure_dose(root: Path, output: Path) -> None:
    tables = root / "tables"
    followup = load(root / "followup.json") or {}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    leace = pairs(pd.read_csv(tables / "leace_runs.csv"), "rank")
    dose_panel(axes[0], leace, "rank", lambda v: f"r={v:g}", style.BLUE, followup.get("F5_manipulation_checks", {}),
               f"LEACE erasure\nF5 {fmt(followup.get('F5'))}")
    normal = pairs(pd.read_csv(tables / "normalization_recoverability.csv"), "strength")
    dose_panel(axes[1], normal, "strength", lambda v: f"s={v:g}", style.AQUA, followup.get("F6_regime_secondary", {}),
               f"Stain normalisation\nF6 {fmt(followup.get('F6'))}")
    linear = pairs(pd.read_csv(tables / "linadv_recoverability.csv"), "adversary_lambda")
    dose_panel(axes[2], linear, "adversary_lambda", lambda v: f"λ={v:g}", style.ORANGE,
               followup.get("F7_manipulation_checks", {}), f"Linear adversary\nF7 {fmt(followup.get('F7'))}")
    axes[0].set_ylabel("P1 minus P2 (pp)")
    figure.tight_layout(w_pad=1.2)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/study2/followup")
    parser.add_argument("--output", default="paper/v2/figures")
    args = parser.parse_args()
    style.apply()
    root, output = Path(args.root), Path(args.output)
    for name, builder in (("carriers", figure_carriers), ("dose", figure_dose)):
        builder(root, output / f"{name}.pdf")
        print(output / f"{name}.pdf")


if __name__ == "__main__":
    main()
