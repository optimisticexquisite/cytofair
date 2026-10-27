#!/usr/bin/env python3
"""Figures for the ISBI 2027 paper, drawn at their printed size from the committed results.

* ``results.pdf`` (full width): the protocol effect on SIPaKMeD per architecture (H1),
  CRIC against SIPaKMeD (H2), and the matched-size factorial cell means (H3).
* ``diagnostics.pdf`` (full width): gap against parent recoverability (F5, F7, F8),
  the leak effect under colour and background interventions (F2, F4), and Grad-CAM
  attribution outside the cell for leak and clean models.

Panels are placed in inches so that tick labels never collide across panels.
Run from the repository root:  python3 paper/isbi2027/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results/study2/analysis"
FOLLOWUP = ROOT / "results/study2/followup"
OUT = Path(__file__).resolve().parent / "figures"

TEXT_WIDTH = 7.008  # spconf text block: 178 mm
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, GREY = "#e6e5df", "#c3c2b7", "#b4b3ad"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LEAK, CLEAN = ORANGE, BLUE

NAMES = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNet-V2", "vit_b_16": "ViT-B/16",
    "convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T", "regnet_y_1_6gf": "RegNetY-1.6GF",
    "efficientnet_v2_s": "EfficientNetV2-S", "maxvit_t": "MaxViT-T", "dinov2_vits14": "DINOv2 ViT-S/14",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.3,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False, "lines.solid_capstyle": "round",
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def t_interval(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


class Canvas:
    """A figure whose axes are placed by their left edge and size in inches."""

    def __init__(self, height: float, bottom: float = 0.36, top: float = 0.24):
        self.fig = plt.figure(figsize=(TEXT_WIDTH, height))
        self.height, self.bottom, self.top = height, bottom, top

    def axes(self, left: float, width: float, *, title_left: float, letter: str, title: str, height=None):
        h = height or self.height - self.bottom - self.top
        ax = self.fig.add_axes([left / TEXT_WIDTH, self.bottom / self.height, width / TEXT_WIDTH, h / self.height])
        self.fig.text(title_left / TEXT_WIDTH, 1 - 0.13 / self.height, f"({letter})", fontsize=8.5,
                      fontweight="bold", color=INK, va="center", ha="left")
        self.fig.text((title_left + 0.24) / TEXT_WIDTH, 1 - 0.13 / self.height, title, fontsize=8.5,
                      color=INK, va="center", ha="left")
        return ax

    def save(self, name: str) -> None:
        self.fig.savefig(OUT / f"{name}.pdf")
        self.fig.savefig(OUT / f"{name}.png", dpi=220)
        plt.close(self.fig)


def zero(ax, orientation="v"):
    (ax.axvline if orientation == "v" else ax.axhline)(0, color=BASELINE, linewidth=0.8, zorder=1)


def dot(ax, x, y, color, *, size=20, marker="o", zorder=4):
    ax.scatter(x, y, s=size, color=color, marker=marker, edgecolors="white", linewidths=0.8, zorder=zorder)


def interval(ax, low, high, y, color, width=1.3):
    ax.plot([low, high], [y, y], color=color, linewidth=width, zorder=2, solid_capstyle="round")


def estimate(ax, record, y, color):
    """A registered model estimate: diamond and 95% interval."""
    ax.plot(record["ci95"], [y, y], color=color, linewidth=1.7, zorder=2, solid_capstyle="butt")
    dot(ax, [record["estimate"]], [y], color, size=34, marker="D")


def style_rows(ax, labels, *, bold=()):
    ax.set_yticks(range(len(labels)), labels)
    for tick, label in zip(ax.get_yticklabels(), labels):
        if label in bold:
            tick.set_color(INK)
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(labels) - 0.45, -0.6)
    ax.grid(axis="y", visible=False)


def protocol_gaps(path: Path, a: str = "P1", b: str = "P2") -> pd.DataFrame:
    runs = pd.read_csv(path)
    wide = runs.pivot_table(index=["architecture", "seed"], columns="protocol", values="accuracy")
    return (100 * (wide[a] - wide[b])).rename("gap").reset_index()


def per_architecture(path: Path):
    gaps = protocol_gaps(path)
    return sorted(((NAMES[a], *t_interval(g.gap)) for a, g in gaps.groupby("architecture")), key=lambda r: -r[1])


# ---------------------------------------------------------------------------- results
def results_figure() -> None:
    canvas = Canvas(height=2.3)
    h1 = load(ANALYSIS / "h1_accuracy.json")
    h1_record = {"estimate": h1["primary_estimate"], "ci95": h1["primary_ci95"]}
    h2 = load(ANALYSIS / "h2_cric_accuracy.json")
    h2_record = {"estimate": h2["primary_estimate"], "ci95": h2["primary_ci95"]}

    # (a) SIPaKMeD: per architecture, and the registered pooled estimate (cluster bootstrap).
    ax = canvas.axes(0.93, 1.27, title_left=0.0, letter="a", title="SIPaKMeD: crop vs. field split")
    rows = per_architecture(ANALYSIS / "runs/phase1_runs.csv")
    estimate(ax, h1_record, 0, INK)
    for y, (_, mean, low, high) in enumerate(rows, start=1):
        interval(ax, low, high, y, LEAK)
        dot(ax, [mean], [y], LEAK)
    style_rows(ax, ["Pooled (H1)"] + [r[0] for r in rows], bold=("Pooled (H1)",))
    ax.axhline(0.5, color=GRID, linewidth=0.6)
    zero(ax)
    ax.set_xlim(-0.3, 2.5)
    ax.set_xlabel("Accuracy gain (pp)")

    # (b) CRIC: per architecture, with both pooled estimates on one axis.
    ax = canvas.axes(3.10, 1.12, title_left=2.32, letter="b", title="CRIC: crop vs. slide split")
    rows = per_architecture(ANALYSIS / "runs/cric_runs.csv")
    estimate(ax, h2_record, 0, INK)
    estimate(ax, h1_record, 1, MUTED)
    for y, (_, mean, low, high) in enumerate(rows, start=2):
        interval(ax, low, high, y, LEAK)
        dot(ax, [mean], [y], LEAK)
    style_rows(ax, ["CRIC (H2)", "SIPaKMeD (H1)"] + [r[0] for r in rows], bold=("CRIC (H2)",))
    ax.axhline(1.5, color=GRID, linewidth=0.6)
    zero(ax)
    ax.set_xlim(-0.8, 15.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Accuracy gain (pp)")

    # (c) The matched-size factorial on SIPaKMeD: accuracy on the common test set per cell.
    ax = canvas.axes(4.98, 1.72, title_left=4.40, letter="c", title="Matched-size factorial (SIPaKMeD)")
    mech = load(ANALYSIS / "mechanism.json")
    fact = pd.read_csv(ANALYSIS / "runs/factorial_runs.csv")
    cells = fact.pivot_table(index="architecture", columns="protocol", values="accuracy") * 100
    xs = [0, 1]  # 504 and 720 distinct training fields
    for _, row in cells.iterrows():
        ax.plot(xs, [row.leak_low, row.leak_high], color=LEAK, alpha=0.35, linewidth=0.8, zorder=2)
        ax.plot(xs, [row.clean_low, row.clean_high], color=CLEAN, alpha=0.35, linewidth=0.8, zorder=2)
    means = cells.mean()
    for kind, color in (("leak", LEAK), ("clean", CLEAN)):
        ys = [means[f"{kind}_low"], means[f"{kind}_high"]]
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3)
        dot(ax, xs, ys, color, size=30, zorder=4)
        ax.text(-0.07, ys[0], kind, color=INK_2, fontsize=7.5, ha="right", va="center")
    h3, div = mech["H3"], mech["diversity_effect_clean_secondary"]

    def fmt(record, signed=True):
        text = f"{record['estimate']:+.2f}" if signed else f"{record['estimate']:.2f}"
        text = "0.00" if text in ("+0.00", "-0.00") else text
        return f"{text} [{record['ci95'][0]:.2f}, {record['ci95'][1]:.2f}]".replace("-", "−")

    mid_leak, mid_clean = means[["leak_low", "leak_high"]].mean(), means[["clean_low", "clean_high"]].mean()
    ax.annotate("", xy=(0.5, mid_leak - 0.07), xytext=(0.5, mid_clean + 0.07),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=0),
                zorder=5)
    ax.text(0.54, (mid_leak + mid_clean) / 2, f"leakage (H3)\n{fmt(h3)}", fontsize=7.3, color=INK,
            ha="left", va="center", linespacing=1.05, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(0.5, 93.62, f"diversity (clean): {fmt(div)}", fontsize=7.3, color=INK_2, ha="center", va="center")
    ax.set_xlim(-0.12, 1.12)
    ax.set_xticks(xs, ["504", "720"])
    ax.set_ylim(93.35, 97.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Distinct training fields")
    ax.set_ylabel("Accuracy on common test set (%)", labelpad=2)
    ax.spines["right"].set_visible(False)
    canvas.save("results")


# ------------------------------------------------------------------------ diagnostics
def recoverability_pairs(path: Path, level: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.assign(value=100 * frame.accuracy, R=100 * frame.within_class_accuracy)
    wide = frame.pivot_table(index=["architecture", "seed", level], columns="protocol", values=["value", "R"])
    return pd.DataFrame({"gap": wide[("value", "P1")] - wide[("value", "P2")],
                         "R": (wide[("R", "P1")] + wide[("R", "P2")]) / 2}).reset_index()


def arrows(ax, frame, level, checks, color):
    """One arrow per architecture: unmanipulated to strongest in-regime level (mean over seeds)."""
    for arch, group in frame.groupby("architecture"):
        check = checks[arch]
        if not check["manipulation_check_passed"]:
            continue
        means = group.groupby(level)[["R", "gap"]].mean()
        start, end = means.loc[0], means.loc[check["strongest"]]
        ax.annotate("", xy=(end.R, end.gap), xytext=(start.R, start.gap),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, alpha=0.85,
                                    mutation_scale=7, shrinkA=0, shrinkB=0), zorder=3)
        dot(ax, [start.R], [start.gap], color, size=9, zorder=4)


def diagnostics_figure() -> None:
    canvas = Canvas(height=2.05)
    follow = load(FOLLOWUP / "followup.json")
    tables = FOLLOWUP / "tables"

    # (a) Decodable is not used: each arrow removes identity; the gap stays put.
    ax = canvas.axes(0.42, 1.78, title_left=0.0, letter="a", title="Erasing parent identity")
    leace = recoverability_pairs(tables / "leace_runs.csv", "rank")
    arrows(ax, leace, "rank", follow["F5_manipulation_checks"], BLUE)
    linear = recoverability_pairs(tables / "linadv_recoverability.csv", "adversary_lambda")
    arrows(ax, linear, "adversary_lambda", follow["F7_manipulation_checks"], ORANGE)
    floor = follow["F5_manipulation_checks"]["resnet18"]["within_class_floor"]  # already in %
    ax.axvline(floor, color=BASELINE, linewidth=0.8, zorder=1)
    ax.text(floor + 0.6, 2.42, "class-only\nfloor", fontsize=6.6, color=MUTED, va="top", linespacing=0.95)
    zero(ax, "h")
    ax.set_xlim(0, 33)
    ax.set_ylim(-0.25, 2.45)
    ax.set_xlabel("Parent recoverability R (%)")
    ax.set_ylabel("P1 − P2 gap (pp)", labelpad=2)
    ax.plot([], [], color=BLUE, lw=1.2, label="LEACE (12 architectures)")
    ax.plot([], [], color=ORANGE, lw=1.2, label="linear adversary (3)")
    ax.legend(loc="lower right", handlelength=1.3, borderaxespad=0.1, bbox_to_anchor=(1.02, -0.01))
    f8 = follow["F8"]
    slope = f"{f8['estimate']:.3f}".replace("-0.000", "0.000")
    ax.text(32.5, 1.62, f"pooled slope (F8)\n{slope} [{f8['ci95'][0]:.3f}, {f8['ci95'][1]:.3f}]"
            .replace("-", "−"), fontsize=6.8, color=INK_2, ha="right", va="center", linespacing=1.0)

    # (b) Interventions: leak minus clean accuracy on the common test set.
    ax = canvas.axes(3.66, 1.18, title_left=2.42, letter="b", title="Leak effect under interventions")
    ablation = pd.read_csv(tables / "ablation_runs.csv")
    carrier = pd.read_csv(tables / "carrier_runs.csv")

    def effects(frame, column):
        frame = frame.assign(div=frame.cell.str.split("_").str[1], leak=frame.cell.str.startswith("leak"))
        wide = frame.pivot_table(index=["architecture", "seed", "div", column], columns="leak", values="accuracy")
        return (100 * (wide[True] - wide[False])).rename("effect").reset_index()

    abl, car = effects(ablation, "ablation"), effects(carrier, "condition")
    entries = [
        ("Test: unmodified", abl.loc[abl.ablation == "original", "effect"], LEAK),
        ("Test: background grey", abl.loc[abl.ablation == "background_grey", "effect"], LEAK),
        ("Test: flat illumination", abl.loc[abl.ablation == "illumination_flat", "effect"], LEAK),
        ("Test: colour transfer", abl.loc[abl.ablation == "stain_transfer", "effect"], CLEAN),
        ("Train: standard", car.loc[car.condition == "standard", "effect"], LEAK),
        ("Train: HED jitter", car.loc[car.condition == "colour", "effect"], LEAK),
    ]
    for y, (_, values, color) in enumerate(entries):
        mean, low, high = t_interval(values)
        interval(ax, low, high, y, color)
        dot(ax, [mean], [y], color, size=22)
    style_rows(ax, [e[0] for e in entries], bold=("Test: colour transfer",))
    ax.axhline(3.5, color=GRID, linewidth=0.6)
    zero(ax)
    ax.set_xlim(-0.5, 2.6)
    ax.set_xlabel("Leak − clean accuracy (pp)")

    # (c) Saliency is not use: Grad-CAM attribution outside the cell mask.
    ax = canvas.axes(5.52, 1.45, title_left=5.10, letter="c", title="Grad-CAM outside the cell")
    cam = pd.read_csv(tables / "gradcam_runs.csv")
    cam = cam.assign(div=cam.cell.str.split("_").str[1], leak=cam.cell.str.startswith("leak"),
                     value=100 * cam.background_attribution_fraction)
    wide = cam.pivot_table(index=["architecture", "seed", "div"], columns="leak", values="value").reset_index()
    markers = {"resnet50": "o", "densenet121": "s", "convnext_tiny": "^"}
    lims = (8, 34)
    ax.plot(lims, lims, color=BASELINE, linewidth=0.8, zorder=1)
    for arch in ("resnet50", "densenet121", "convnext_tiny"):
        group = wide.loc[wide.architecture == arch]
        ax.scatter(group[False], group[True], s=12, marker=markers[arch], color=INK_2, alpha=0.8,
                   edgecolors="white", linewidths=0.4, zorder=3, label=NAMES[arch])
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xticks([10, 20, 30])
    ax.set_yticks([10, 20, 30])
    ax.set_xlabel("Clean models (%)")
    ax.set_ylabel("Leak models (%)", labelpad=2)
    ax.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    gc = follow["gradcam_descriptive"]["leak_effect_on_background_attribution"]
    ax.text(8.8, 33, f"leak − clean\n{gc['estimate']:.2f} [{gc['ci95'][0]:.2f}, {gc['ci95'][1]:.2f}] pp"
            .replace("-", "−"), fontsize=6.8, color=INK_2, ha="left", va="top", linespacing=1.0)
    canvas.save("diagnostics")


def main() -> None:
    apply_style()
    OUT.mkdir(exist_ok=True)
    results_figure()
    diagnostics_figure()
    print(OUT / "results.pdf", OUT / "diagnostics.pdf")


if __name__ == "__main__":
    main()
