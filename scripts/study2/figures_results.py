#!/usr/bin/env python3
"""Study 2 result figures, drawn only from the registered analysis outputs.

* ``h1.pdf``: per-architecture P1 - P2 (primary) and P2 - P3 (secondary) with
  the registered pooled estimate.
* ``factorial.pdf``: leakage effect at matched diversity and the clean
  diversity effect, per architecture and pooled.
* ``transplant.pdf``: accuracy under each background condition for leak and
  clean models, and the context-route contrast.
* ``grl.pdf``: fingerprint recoverability, P1 - P2 gap, and their relation
  across the gradient-reversal sweep.
* ``cric.pdf``: per-architecture CRIC P1 - P2 with both pooled estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import figstyle as style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

NAMES = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNet-V2", "vit_b_16": "ViT-B/16",
    "convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T", "regnet_y_1_6gf": "RegNetY-1.6GF",
    "efficientnet_v2_s": "EfficientNetV2-S", "maxvit_t": "MaxViT-T", "dinov2_vits14": "DINOv2 ViT-S/14",
}


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def t_interval(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if len(values) < 2:
        return mean, mean, mean
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def forest(ax, rows, pooled, *, color, pooled_label, xlabel, pooled_color=style.ORANGE):
    """Per-row mean and 95% CI, then the registered pooled estimate as a diamond."""
    rows = sorted(rows, key=lambda row: row[1])
    for position, (label, mean, low, high) in enumerate(rows):
        ax.plot([low, high], [position, position], color=color, linewidth=1.3, zorder=2)
        style.ring(ax, [mean], [position], color, size=22)
    labels = [label for label, *_ in rows]
    if pooled is not None:
        position = -1.4
        mean, low, high = pooled
        ax.plot([low, high], [position, position], color=pooled_color, linewidth=1.6, zorder=2)
        style.ring(ax, [mean], [position], pooled_color, size=48, marker="D")
        ax.set_yticks([*range(len(rows)), position], [*labels, pooled_label])
        ax.get_yticklabels()[-1].set_color(style.INK)
    else:
        ax.set_yticks(range(len(rows)), labels)
    style.zero_line(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def figure_h1(analysis: Path, output: Path) -> None:
    runs = pd.read_csv(analysis / "runs" / "phase1_runs.csv")
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), sharey=False)
    for ax, (left, right, file, color, pooled_color, title) in zip(
        axes,
        (("P1", "P2", "h1_accuracy.json", style.BLUE, style.ORANGE, "Crop-level minus parent-level (H1)"),
         ("P2", "P3", "p2_p3_accuracy.json", style.GRAY_SERIES, style.INK_2, "Parent-level minus component-level")),
        strict=True,
    ):
        wide = runs.loc[runs.protocol.isin([left, right])].pivot_table(
            index=["architecture", "seed"], columns="protocol", values="accuracy")
        differences = (100 * (wide[left] - wide[right])).groupby("architecture")
        rows = [(NAMES.get(arch, arch), *t_interval(values)) for arch, values in differences]
        record = load(analysis / file)
        pooled = None if record is None else (record["primary_estimate"], *record["primary_ci95"])
        forest(ax, rows, pooled, color=color, pooled_label="All (registered model)",
               xlabel=f"{left} minus {right} accuracy (percentage points)", pooled_color=pooled_color)
        ax.set_title(title, loc="left", color=style.INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(w_pad=2.5)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def figure_factorial(analysis: Path, output: Path) -> None:
    runs = pd.read_csv(analysis / "runs" / "factorial_runs.csv")
    runs["value"] = 100 * runs.accuracy
    wide = runs.pivot_table(index=["architecture", "seed"], columns="protocol", values="value")
    mechanism = load(analysis / "mechanism.json")
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    rows = []
    for arch, group in wide.groupby("architecture"):
        for level, marker in (("high", "o"), ("low", "s")):
            rows.append((arch, level, marker, *t_interval(group[f"leak_{level}"] - group[f"clean_{level}"])))
    ax = axes[0]
    order = sorted({row[0] for row in rows}, key=lambda arch: -np.mean([r[3] for r in rows if r[0] == arch]))
    for index, arch in enumerate(order):
        for offset, (level, marker) in ((-0.15, ("high", "o")), (0.15, ("low", "s"))):
            _, _, _, mean, low, high = next(r for r in rows if r[0] == arch and r[1] == level)
            y = index + offset
            ax.plot([low, high], [y, y], color=style.LEAK, linewidth=1.2, zorder=2)
            style.ring(ax, [mean], [y], style.LEAK, size=20, marker=marker)
    h3 = None if mechanism is None else mechanism["H3"]
    if h3 is not None:
        ax.plot(h3["ci95"], [-1.2, -1.2], color=style.INK_2, linewidth=1.6)
        style.ring(ax, [h3["estimate"]], [-1.2], style.INK_2, size=44, marker="D")
    ax.set_yticks([*range(len(order)), -1.2], [*[NAMES.get(a, a) for a in order], "All (H3)"])
    style.zero_line(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Leak minus clean accuracy (pp)")
    ax.set_title("Leakage at matched diversity", loc="left", color=style.INK)
    ax.legend(handles=[Line2D([], [], color=style.LEAK, marker="o", linestyle="", markersize=4, label="720 fields"),
                       Line2D([], [], color=style.LEAK, marker="s", linestyle="", markersize=4, label="504 fields")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7)
    ax = axes[1]
    diversity_rows = [(NAMES.get(arch, arch), *t_interval(group["clean_high"] - group["clean_low"]))
                      for arch, group in wide.groupby("architecture")]
    effect = None if mechanism is None else mechanism["diversity_effect_clean_secondary"]
    forest(ax, diversity_rows, None if effect is None else (effect["estimate"], *effect["ci95"]),
           color=style.CLEAN, pooled_label="All (secondary)", xlabel="720 minus 504 fields, clean cells (pp)",
           pooled_color=style.INK_2)
    ax.set_title("Training diversity without leakage", loc="left", color=style.INK)
    figure.tight_layout(w_pad=2.5)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def figure_transplant(analysis: Path, output: Path) -> None:
    table = pd.read_csv(analysis / "mechanism" / "transplant_runs.csv")
    table = table.loc[table.subset == "all"].copy()
    table["leak"] = table.cell.str.startswith("leak")
    conditions = [("acc_original", "original"), ("acc_self", "self"), ("acc_sibling", "sibling"),
                  ("acc_unseen_same_class", "unseen\nsame class"), ("acc_unseen_other_class", "unseen\nother class")]
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for leak, color, shift, label in ((False, style.CLEAN, -0.12, "clean models"), (True, style.LEAK, 0.12, "leak models")):
        subset = table.loc[table.leak == leak]
        means = []
        for index, (column, _) in enumerate(conditions):
            mean, low, high = t_interval(100 * subset[column])
            means.append(mean)
            ax.plot([index + shift] * 2, [low, high], color=color, linewidth=1.2, zorder=2)
        ax.plot(np.arange(len(conditions)) + shift, means, color=color, linewidth=1.0, alpha=0.6, zorder=1)
        style.ring(ax, np.arange(len(conditions)) + shift, means, color, size=22, label=label)
    ax.set_xticks(range(len(conditions)), [name for _, name in conditions])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Accuracy on composites (%)")
    ax.set_title("Background transplants", loc="left", color=style.INK)
    ax.legend(loc="lower left", fontsize=7)
    ax = axes[1]
    table["route"] = 100 * (table.acc_sibling - table.acc_unseen_same_class)
    for leak, color, y in ((False, style.CLEAN, 0), (True, style.LEAK, 1)):
        values = table.loc[table.leak == leak, "route"]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(values))
        ax.scatter(values, y + jitter, s=7, color=color, alpha=0.35, linewidths=0, zorder=2)
        mean, low, high = t_interval(values)
        ax.plot([low, high], [y, y], color=style.INK_2, linewidth=1.4, zorder=3)
        style.ring(ax, [mean], [y], color, size=40, zorder=4)
    mechanism = load(analysis / "mechanism.json")
    title = "Own-context route"
    if mechanism is not None:
        h4 = mechanism["H4"]
        title += f"\nH4 {h4['estimate']:.2f} pp [{h4['ci95'][0]:.2f}, {h4['ci95'][1]:.2f}]".replace("-", "\u2212")
    ax.set_yticks([0, 1], ["clean models", "leak models"])
    style.zero_line(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sibling minus unseen same-class (pp)")
    ax.set_title(title, loc="left", color=style.INK)
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def figure_grl(analysis: Path, output: Path) -> None:
    grl = pd.read_csv(analysis / "mechanism" / "grl_recoverability.csv")
    grl["R"], grl["value"] = 100 * grl.within_class_accuracy, 100 * grl.accuracy
    lambdas = sorted(grl.adversary_lambda.unique())
    positions = {value: index for index, value in enumerate(lambdas)}
    architectures = sorted(grl.architecture.unique())
    colors = dict(zip(architectures, (style.BLUE, style.AQUA), strict=False))
    dodge = dict(zip(architectures, (-0.12, 0.12), strict=False))
    mechanism = load(analysis / "mechanism.json")
    checks = {} if mechanism is None else mechanism.get("H5_manipulation_checks", {})
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for arch in architectures:
        subset = grl.loc[grl.architecture == arch]
        r = subset.groupby("adversary_lambda").R.mean()
        x = [positions[v] for v in r.index]
        axes[0].plot(x, r.values, color=colors[arch], zorder=2)
        style.ring(axes[0], x, r.values, colors[arch], size=16, label=NAMES.get(arch, arch))
        wide = subset.pivot_table(index=["seed", "adversary_lambda"], columns="protocol", values="value")
        gap = (wide.P1 - wide.P2).groupby("adversary_lambda")
        rows = [(positions[v] + dodge[arch], *t_interval(values)) for v, values in gap]
        for x_value, _, low, high in rows:
            axes[1].plot([x_value, x_value], [low, high], color=colors[arch], linewidth=1.0)
        axes[1].plot([row[0] for row in rows], [row[1] for row in rows], color=colors[arch], zorder=2)
        style.ring(axes[1], [row[0] for row in rows], [row[1] for row in rows], colors[arch], size=16)
        r_pair = subset.pivot_table(index=["seed", "adversary_lambda"], columns="protocol", values="R").mean(axis=1)
        gap_pairs = (wide.P1 - wide.P2).reindex(r_pair.index)
        regime = set(checks.get(arch, {}).get("non_degenerate_lambdas", lambdas))
        # Hollow grey: outside the regime, or an architecture that failed the manipulation check.
        passed = checks.get(arch, {}).get("manipulation_check_passed", True)
        inside = np.array([passed and value in regime for value in r_pair.index.get_level_values("adversary_lambda")])
        axes[2].scatter(r_pair.values[inside], gap_pairs.values[inside], s=9, color=colors[arch], alpha=0.55,
                        linewidths=0, zorder=2)
        axes[2].scatter(r_pair.values[~inside], gap_pairs.values[~inside], s=9, facecolors="none",
                        edgecolors=style.GRAY_SERIES, linewidths=0.6, zorder=1)
    floor = 100 * grl.within_class_floor.mean()
    axes[0].axhline(floor, color=style.MUTED, linewidth=0.8)
    axes[0].annotate("class-only floor", (0, floor), xytext=(0, 3), textcoords="offset points", fontsize=7,
                     color=style.INK_2)
    for ax in axes[:2]:
        ax.set_xticks(range(len(lambdas)), [f"{v:g}".removeprefix("0") if 0 < v < 1 else f"{v:g}" for v in lambdas])
        ax.set_xlabel("Reversal strength $\\lambda$")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Within-class parent ID (%)")
    axes[0].set_title("Fingerprint recoverability", loc="left", color=style.INK)
    axes[1].set_ylabel("P1 minus P2 (pp)")
    axes[1].set_title("Inflation", loc="left", color=style.INK)
    style.zero_line(axes[1], "horizontal")
    style.zero_line(axes[2], "horizontal")
    h5 = None if mechanism is None else mechanism.get("H5", {})
    subtitle = ""
    if h5 and h5.get("estimate") is not None:
        subtitle = f"\nH5 slope {h5['estimate']:.3f} [{h5['ci95'][0]:.3f}, {h5['ci95'][1]:.3f}]".replace("-", "\u2212")
    elif h5 and h5.get("status"):
        subtitle = "\nH5 untestable"
    axes[2].set_xlabel("Recoverability (%)")
    axes[2].set_ylabel("P1 minus P2 (pp)")
    axes[2].set_title("Inflation vs fingerprint" + subtitle, loc="left", color=style.INK)
    axes[0].legend(loc="upper left", fontsize=7)
    figure.tight_layout(w_pad=1.6)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def figure_cric(analysis: Path, output: Path) -> None:
    runs = pd.read_csv(analysis / "runs" / "cric_runs.csv")
    wide = runs.loc[runs.protocol.isin(["P1", "P2"])].pivot_table(
        index=["architecture", "seed"], columns="protocol", values="accuracy")
    rows = [(NAMES.get(arch, arch), *t_interval(100 * (group.P1 - group.P2))) for arch, group in wide.groupby("architecture")]
    cric, sipakmed = load(analysis / "h2_cric_accuracy.json"), load(analysis / "h1_accuracy.json")
    figure, ax = plt.subplots(figsize=(4.4, 2.7))
    forest(ax, rows, None if cric is None else (cric["primary_estimate"], *cric["primary_ci95"]),
           color=style.BLUE, pooled_label="CRIC (H2)", xlabel="P1 minus P2 accuracy (pp)")
    if sipakmed is not None:
        y = -2.4
        ax.plot(sipakmed["primary_ci95"], [y, y], color=style.GRAY_SERIES, linewidth=1.6)
        style.ring(ax, [sipakmed["primary_estimate"]], [y], style.GRAY_SERIES, size=44, marker="D")
        ticks, labels = list(ax.get_yticks()), [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticks([*ticks, y], [*labels, "SIPaKMeD (H1)"])
    ax.set_title("CRIC rescore", loc="left", color=style.INK)
    figure.tight_layout()
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="results/study2/analysis")
    parser.add_argument("--output", default="paper/v2/figures")
    parser.add_argument("--only", nargs="*", default=["h1", "factorial", "transplant", "grl", "cric"])
    args = parser.parse_args()
    style.apply()
    analysis, output = Path(args.analysis), Path(args.output)
    builders = {"h1": figure_h1, "factorial": figure_factorial, "transplant": figure_transplant,
                "grl": figure_grl, "cric": figure_cric}
    for name in args.only:
        builders[name](analysis, output / f"{name}.pdf")
        print(output / f"{name}.pdf")


if __name__ == "__main__":
    main()
