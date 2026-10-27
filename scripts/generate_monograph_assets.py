"""Generate evidence-grounded figures and LaTeX tables for the CytoFair monograph.

The script reads only frozen repository artifacts.  It does not train a model or
alter a registered result.  The Herlev image used in the transformation plate is
explicitly a visual demonstration of the exact probe preprocessing code; the
reported probe was run on SIPaKMeD.
"""

from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "monograph" / "assets"
TABLES = ROOT / "monograph" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

BLUE = "#176B87"
TEAL = "#218C74"
RED = "#C4455A"
GOLD = "#D99A2B"
PURPLE = "#735DA5"
INK = "#24323D"
MUTED = "#687884"
PALE = "#EEF4F7"
COLORS = [BLUE, TEAL, RED, GOLD, PURPLE, "#5B8FF9", "#A65F46"]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": "#AAB7C0",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDE5EA",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def read_jsons(pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        record = json.loads(Path(path).read_text())
        if "task" not in record:
            continue
        record["result_file"] = Path(path).name
        rows.append(record)
    return pd.DataFrame(rows)


def box(ax, xy, wh, text, color=BLUE, face=PALE, fontsize=10, lw=1.6):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
        ec=color, fc=face, lw=lw
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)
    return patch


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.5, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=12, connectionstyle=connectionstyle))


def lifecycle_figure():
    events = [
        ("22 Aug\n04:54", "Preregister\nconfirmatory plan", BLUE),
        ("22 Aug\n05:40", "Correct class order\nbefore results", GOLD),
        ("22 Aug\n07:14", "Provenance +\nstructural audit", TEAL),
        ("22 Aug\n13:22", "SIP classifier +\nprobe results", TEAL),
        ("24 Aug\n03:15", "Register Cx22\nsegmentation", BLUE),
        ("24 Aug\n04:02", "FP16 throughput\namendment", GOLD),
        ("28 Aug\n05:53", "MAnet FP32 after\nnon-finite loss", RED),
        ("6 Sep\n23:32", "Instance runtime\ncorrection", GOLD),
        ("7 Sep\n18:14", "175/175 Cx22\ncells complete", TEAL),
        ("13 Sep", "Comprehensive\nreport", PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, len(events) - 1)
    ax.set_ylim(-1.1, 1.25)
    ax.axis("off")
    ax.plot(range(len(events)), [0] * len(events), color="#B7C4CC", lw=3, zorder=0)
    for i, (date, label, color) in enumerate(events):
        side = 1 if i % 2 == 0 else -1
        ax.scatter(i, 0, s=90, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.plot([i, i], [0, side * 0.34], color=color, lw=1.4)
        ax.text(i, side * 0.48, label, ha="center", va="center",
                fontsize=9, color=INK, weight="bold")
        ax.text(i, side * 0.94, date, ha="center", va="center", fontsize=8, color=MUTED)
    ax.set_title("Project lifecycle: decisions were recorded before, during, and after execution",
                 loc="left", fontweight="bold", color=INK)
    save(fig, "lifecycle_timeline.png")


def hierarchy_figure():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.2, 5.2, "The hidden hierarchy", fontsize=16, weight="bold", color=INK)
    box(ax, (0.3, 3.45), (2.1, 1.0), "Source field\n(parent)", TEAL, "#E9F6F2", 12)
    crop_xy = [(3.2, 4.15), (3.2, 2.85), (5.55, 4.15), (5.55, 2.85)]
    for j, (x, y) in enumerate(crop_xy, 1):
        box(ax, (x, y), (1.55, 0.75), f"Cell crop {j}", BLUE, "#EDF5FA", 10)
        arrow(ax, (2.4, 3.95), (x, y + 0.37), color="#8CA4B1")
    ax.text(3.2, 2.15, "Same stain, illumination, focus, texture, artifacts - and sometimes pixels",
            color=MUTED, fontsize=9)

    ax.text(8.0, 5.2, "Why filename-disjoint can still leak", fontsize=16, weight="bold", color=INK)
    box(ax, (8.1, 3.65), (1.9, 0.9), "TRAIN\nCrops 1, 2, 3", BLUE, "#E7F2FA", 11)
    box(ax, (10.8, 3.65), (1.9, 0.9), "TEST\nCrop 4", RED, "#FCECEF", 11)
    arrow(ax, (9.95, 4.1), (10.85, 4.1), color=RED, connectionstyle="arc3,rad=-0.35")
    ax.text(10.4, 4.7, "shared parent", ha="center", color=RED, fontsize=10, weight="bold")
    box(ax, (8.1, 1.55), (4.6, 1.1), "Correct unit: assign the entire parent (or conservative\nnear-duplicate component) to exactly one partition", TEAL, "#E9F6F2", 11)
    ax.text(8.1, 0.85, "Core distinction", fontsize=10, color=MUTED, weight="bold")
    ax.text(8.1, 0.35, "The crop is the network input. The parent is the independent sampling unit.",
            fontsize=11, color=INK)
    save(fig, "hierarchy_and_leakage.png")


def dataset_figures():
    structural = pd.read_csv(ROOT / "results/table1_structural_leakage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    x = np.arange(len(structural))
    axes[0].bar(x - 0.17, structural.crops, width=0.34, color=BLUE, label="released observations")
    axes[0].bar(x + 0.17, structural.parents, width=0.34, color=TEAL, label="parents")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, structural.dataset)
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("Released observations are not always independent")
    axes[0].legend(frameon=False)
    ratio = structural.crops / structural.parents
    bars = axes[1].bar(structural.dataset, ratio, color=COLORS[:4])
    axes[1].set_ylabel("Mean crops per parent")
    axes[1].set_title("Multiplicity determines leakage opportunity")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    save(fig, "dataset_scale.png")

    maps = {
        "SIPaKMeD": ROOT / "provenance/sipakmed_parent_map.parquet",
        "Cx22": ROOT / "provenance/cx22_parent_map.parquet",
        "CRIC": ROOT / "provenance/cric_parent_map.parquet",
        "Herlev": ROOT / "provenance/herlev_parent_map.parquet",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (name, path), color in zip(axes.ravel(), maps.items(), COLORS):
        df = pd.read_parquet(path)
        counts = df.groupby("parent_id").size()
        cap = min(int(counts.max()), 80)
        bins = np.arange(0.5, cap + 1.6, 1)
        clipped = counts.clip(upper=cap)
        ax.hist(clipped, bins=bins, color=color, alpha=0.88, edgecolor="white")
        ax.axvline(counts.mean(), color=INK, ls="--", lw=1.2, label=f"mean {counts.mean():.2f}")
        ax.set_title(f"{name}: {len(counts):,} parents")
        ax.set_xlabel("Crops per parent" + (f" (values >{cap} clipped)" if counts.max() > cap else ""))
        ax.set_ylabel("Parents")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Parent-size distributions explain why the same split rule behaves differently",
                 x=0.04, ha="left", fontsize=15, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "parent_size_distributions.png")


def provenance_figures():
    b = pd.read_parquet(ROOT / "provenance/sipakmed_branch_b.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(b.match_confidence, bins=40, color=TEAL)
    axes[0].set_xlim(0.9999, 1.00001)
    axes[0].set_title("Winning NCC")
    axes[0].set_xlabel("Normalized cross-correlation")
    axes[0].set_ylabel("Crops")
    axes[0].axvline(b.match_confidence.min(), color=RED, ls="--", label=f"min {b.match_confidence.min():.6f}")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(b.match_margin, bins=50, color=BLUE)
    axes[1].axvline(0.02, color=RED, ls="--", label="diagnostic 0.02")
    axes[1].set_title("Winner-runner-up margin")
    axes[1].set_xlabel("NCC margin")
    axes[1].legend(frameon=False, fontsize=8)
    accepted = int(b.accepted.sum())
    axes[2].pie([accepted, len(b) - accepted], colors=[TEAL, GOLD], startangle=90,
                labels=["margin diagnostic pass", "retained despite low margin"],
                autopct=lambda p: f"{p:.1f}%")
    axes[2].set_title("Diagnostic is not the acceptance gate")
    fig.suptitle("SIPaKMeD Branch B: exact content matching used as an independent audit",
                 x=0.04, ha="left", fontsize=15, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "provenance_diagnostics.png")

    # Draw one recovered field using real stored coordinates (without redistributing pixels).
    counts = b.groupby("parent_id").size().sort_values(ascending=False)
    parent = counts.index[0]
    g = b[b.parent_id == parent].copy()
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.add_patch(Rectangle((0, 0), 2048, 1536, fc="#F6F2EE", ec=INK, lw=2))
    for i, row in enumerate(g.itertuples(), 1):
        ax.add_patch(Rectangle((row.x, row.y), row.width, row.height,
                               fill=False, ec=COLORS[(i - 1) % len(COLORS)], lw=2))
        ax.text(row.x + row.width / 2, row.y + row.height / 2, str(i), ha="center", va="center",
                fontsize=8, color=INK, bbox={"boxstyle": "circle,pad=.15", "fc": "white", "ec": "none", "alpha": .8})
    ax.set_xlim(0, 2048)
    ax.set_ylim(1536, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate (pixels)")
    ax.set_ylabel("y coordinate (pixels)")
    ax.set_title(f"Recovered bounding boxes for parent {parent} ({len(g)} crops)\n"
                 "Geometry reconstructed from the Branch-B audit; source pixels are not redistributed")
    save(fig, "recovered_bbox_example.png")


def split_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3))
    specs = [
        ("P1: crop-level", "Sibling crops may straddle", True, False),
        ("P1-prime: size matched", "Same P1 test; training count matched to P2", True, False),
        ("P2: parent-disjoint", "Every parent remains whole", False, False),
        ("P3: component-disjoint", "Near-duplicate-linked parents remain whole", False, True),
    ]
    for ax, (title, subtitle, split_siblings, components) in zip(axes.ravel(), specs):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        ax.add_patch(Rectangle((0.4, 0.6), 4.2, 4.5, fc="#E7F2FA", ec=BLUE, lw=1.5))
        ax.add_patch(Rectangle((5.4, 0.6), 4.2, 4.5, fc="#FCECEF", ec=RED, lw=1.5))
        ax.text(2.5, 5.35, "TRAIN", ha="center", weight="bold", color=BLUE)
        ax.text(7.5, 5.35, "TEST", ha="center", weight="bold", color=RED)
        parents = {"A": TEAL, "B": GOLD, "C": PURPLE}
        coords = [(1.3, 3.6), (2.8, 3.6), (1.3, 2.0), (2.8, 2.0)]
        for i, (x, y) in enumerate(coords):
            c = parents["A" if i < 3 else "B"]
            ax.add_patch(plt.Circle((x, y), .42, fc=c, ec="white", lw=1.2))
        if split_siblings:
            ax.add_patch(plt.Circle((6.7, 3.6), .42, fc=parents["A"], ec="white", lw=1.2))
            ax.add_patch(plt.Circle((8.2, 2.4), .42, fc=parents["C"], ec="white", lw=1.2))
            ax.annotate("A straddles", (6.7, 4.25), ha="center", fontsize=8, color=RED)
        else:
            for x, y in [(6.5, 3.6), (8.0, 3.6), (6.5, 2.0)]:
                ax.add_patch(plt.Circle((x, y), .42, fc=parents["C"], ec="white", lw=1.2))
            if components:
                ax.plot([6.5, 8.0, 6.5], [3.6, 3.6, 2.0], color=PURPLE, lw=2, alpha=.5)
                ax.annotate("one component", (7.4, 4.25), ha="center", fontsize=8, color=PURPLE)
        ax.set_title(title, loc="left", weight="bold", color=INK)
        ax.text(0.45, 0.15, subtitle, fontsize=9, color=MUTED)
    fig.suptitle("Four protocols separate crop count from independence",
                 x=0.05, ha="left", fontsize=16, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "split_protocols.png")


def structural_figures():
    s = pd.read_csv(ROOT / "results/table1_structural_leakage.csv")
    k = np.arange(1, 41)
    p = 1 - np.power(.8, k) - np.power(.2, k)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(k, 100 * p, color=BLUE, lw=3)
    for kval in [1, 2, 4, 8, 16, 32]:
        ax.scatter(kval, 100 * (1 - .8**kval - .2**kval), color=RED, s=35)
        ax.annotate(f"k={kval}", (kval, 100 * (1 - .8**kval - .2**kval)),
                    xytext=(3, -13), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Crops emitted by one parent, k")
    ax.set_ylabel("Probability parent appears in both train and test (%)")
    ax.set_ylim(-2, 102)
    ax.set_title(r"Analytic mechanism: $P(\mathrm{straddle}\mid k)=1-0.8^k-0.2^k$", loc="left", weight="bold")
    ax.text(20, 25, "At k=1 straddling is impossible.\nAs k grows, straddling becomes nearly certain.",
            color=MUTED, fontsize=10, bbox={"fc": "white", "ec": "#D7E0E5", "boxstyle": "round,pad=.4"})
    save(fig, "straddle_probability.png")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(s)); w = .26
    ax.bar(x - w, s.analytic_p1_straddle_pct, w, color=GOLD, label="analytic P1 straddle")
    obs = s.observed_p1_straddle_pct_mean_sd.str.extract(r"([\d.]+) ± ([\d.]+)").astype(float)
    leak = s.observed_p1_test_leak_pct_mean_sd.str.extract(r"([\d.]+) ± ([\d.]+)").astype(float)
    ax.bar(x, obs[0], w, yerr=obs[1], capsize=3, color=BLUE, label="observed P1 straddle")
    ax.bar(x + w, leak[0], w, yerr=leak[1], capsize=3, color=RED, label="P1 test crops with train parent")
    ax.set_xticks(x, s.dataset)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 108)
    ax.set_title("Structural leakage appears where parent multiplicity exists and vanishes under P2/P3",
                 loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, "structural_results.png")

    seed = pd.read_csv(ROOT / "results/table1_seed_level.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in seed.groupby("dataset"):
        axes[0].plot(g.seed.astype(str), 100 * g.p1_straddle_fraction, marker="o", label=name)
        axes[1].plot(g.seed.astype(str), 100 * g.p1_leaked_test_crop_fraction, marker="o", label=name)
    axes[0].set_title("Parent straddle across fixed seeds")
    axes[1].set_title("Test crops whose parent is in training")
    for ax in axes:
        ax.set_xlabel("Seed"); ax.set_ylabel("Percent"); ax.set_ylim(-2, 103)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("The five-seed spread is visible rather than hidden by a single split",
                 x=.04, ha="left", fontsize=15, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, "structural_seed_spread.png")


def overlap_and_duplicate_figures():
    p = pd.read_csv(ROOT / "results/pixel_overlap_seed_level.csv")
    p = p[p.protocol == "P1"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(p)); w=.36
    ax.bar(x-w/2, 100*p.overlapping_cross_split_pair_fraction, w, color=BLUE,
           label="cross-split sibling pairs sharing pixels")
    ax.bar(x+w/2, 100*p.test_crop_pixel_overlap_fraction, w, color=RED,
           label="test crops sharing pixels with training")
    ax.set_xticks(x, p.seed)
    ax.set_xlabel("Fixed seed")
    ax.set_ylabel("Percent")
    ax.set_title("Literal pixel overlap under P1 (both measures are exactly zero under P2/P3)",
                 loc="left", weight="bold")
    ax.legend(frameon=False)
    save(fig, "literal_pixel_overlap.png")

    d = pd.read_csv(ROOT / "results/duplicates/sipakmed_near_duplicate_pairs.csv")
    m = pd.read_parquet(ROOT / "provenance/sipakmed_parent_map_p3.parquet")
    sizes = m[["parent_id", "p3_group_id"]].drop_duplicates().groupby("p3_group_id").size()
    linked = sizes[sizes > 1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].scatter(d.phash_distance, d.dhash_distance, s=18, alpha=.65, color=PURPLE)
    axes[0].axvline(6, color=RED, ls="--"); axes[0].axhline(6, color=RED, ls="--")
    axes[0].set_xlabel("pHash Hamming distance"); axes[0].set_ylabel("dHash Hamming distance")
    axes[0].set_title(f"{len(d)} cross-parent candidate pairs")
    cnt = linked.value_counts().sort_index()
    axes[1].bar(cnt.index.astype(str), cnt.values, color=TEAL)
    axes[1].set_xlabel("Parents in connected component"); axes[1].set_ylabel("Components")
    axes[1].set_title(f"{len(linked)} linked components; largest={linked.max()}")
    fig.suptitle("P3 is a conservative sensitivity grouping, not a claim that every candidate is a duplicate",
                 x=.04, ha="left", fontsize=14, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, "duplicate_audit.png")


def transform_plate():
    source = ROOT / "data/extracted/herlev/carcinoma_in_situ/1.BMP"
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        source = next((ROOT / "data/extracted/herlev").rglob("*.BMP"))
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat_t, _ = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lum_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_mask = ((hsv[..., 1] > sat_t) | (gray < lum_t)).astype(np.uint8)
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    telea = cv2.inpaint(rgb, closed, 5, cv2.INPAINT_TELEA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_mean = np.asarray((190., 140., 135.), np.float32)
    target_sd = np.asarray((35., 14., 14.), np.float32)
    sm = lab.reshape(-1,3).mean(0); ss = lab.reshape(-1,3).std(0)
    stain = cv2.cvtColor(np.clip((lab-sm)*(target_sd/np.maximum(ss,1.0))+target_mean,0,255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(rgb, (0,0), sigmaX=3)
    highpass = np.clip(rgb.astype(np.int16)-blur.astype(np.int16)+128,0,255).astype(np.uint8)
    gray3 = np.repeat(gray[...,None],3,axis=2)
    sat = hsv[...,1]
    lum = gray
    panels = [
        (rgb, "A  Original RGB"),
        (gray3, "B  Grayscale"),
        (stain, "C  LAB stain-normalized"),
        (sat, f"D  Saturation channel\nOtsu threshold={sat_t:.0f}"),
        (lum, f"E  Luminance channel\nOtsu threshold={lum_t:.0f}"),
        (raw_mask * 255, "F  OR-combined Otsu mask"),
        (closed * 255, "G  5x5 morphological close"),
        (telea, "H  Telea-inpainted background"),
        (highpass, "I  High-pass residual"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.5))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray" if img.ndim==2 else None, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.axis("off")
    fig.suptitle("Exact probe preprocessing, visualized step by step on a local Herlev cell",
                 x=.04, ha="left", fontsize=16, weight="bold", color=INK)
    fig.text(.04,.01,
             "Demonstration image only: the reported parent-identity experiment used SIPaKMeD. "
             "Panels D-G expose the mask construction used before Telea inpainting.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, .035, 1, .95))
    save(fig, "probe_transformations.png")


def probe_figures():
    p = pd.read_csv(ROOT / "results/parent_probe_summary.csv")
    display = {
        "rgb": "RGB",
        "grayscale": "Grayscale",
        "stain_normalized": "Stain Normalized",
        "background_only": "Background Only",
        "highpass": "Highpass",
    }
    p["label"] = p.image_mode.map(display)
    p.loc[p.shuffled_labels, "label"] = "RGB shuffled labels"
    order = ["RGB", "Grayscale", "Stain Normalized", "Background Only", "Highpass", "RGB shuffled labels"]
    p["label"] = pd.Categorical(p.label, order, ordered=True)
    p = p.sort_values("label")
    fig, ax = plt.subplots(figsize=(11, 5.1))
    bars=ax.bar(p.label.astype(str), 100*p["mean"], yerr=100*p.sd, capsize=4,
            color=[TEAL, BLUE, GOLD, PURPLE, RED, "#AAB7C0"])
    chance=100/424
    ax.axhline(chance, color=INK, ls="--", label=f"exact chance = {chance:.4f}%")
    ax.set_ylabel("Held-out parent-identification accuracy (%)")
    ax.set_title("Parent identity remains recoverable after every prespecified transformation",
                 loc="left", weight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, labels=[f"{100*v:.2f}%" for v in p["mean"]], padding=3, fontsize=8)
    ax.legend(frameon=False)
    save(fig, "probe_results.png")

    runs=read_jsons("results/probe/*.json")
    fig, axes=plt.subplots(1,2,figsize=(12,4.7))
    for mode,g in runs[~runs.shuffled_labels].groupby("image_mode"):
        histories=[]
        for h in g.history:
            histories.append([e["test_accuracy"] for e in h])
        a=np.array(histories)
        axes[0].plot(np.arange(1,a.shape[1]+1),100*a.mean(0),label=mode.replace("_"," "))
    sg=runs[runs.shuffled_labels]
    a=np.array([[e["test_accuracy"] for e in h] for h in sg.history])
    axes[0].plot(np.arange(1,a.shape[1]+1),100*a.mean(0),label="shuffled",color=INK,ls="--")
    axes[0].axhline(100/424,color=MUTED,ls=":")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Mean held-out accuracy")
    rgb=runs[(runs.image_mode=="rgb") & (~runs.shuffled_labels)]
    for _,r in rgb.iterrows():
        axes[1].plot([e["epoch"] for e in r.history],[e["test_accuracy"]*100 for e in r.history],alpha=.75,label=f"seed {r.seed}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RGB test accuracy (%)")
    axes[1].set_title("RGB trajectory for every seed")
    axes[0].legend(frameon=False,fontsize=8,ncol=2); axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle("Probe learning curves: a genuine parent signal learns; shuffled labels do not",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig,"probe_learning_curves.png")


def classification_figures():
    summary=pd.read_csv(ROOT/"results/classification_summary.csv")
    acc=summary[summary.metric=="accuracy"].copy()
    arch_order=["densenet121","efficientnet_b0","mobilenet_v2","resnet18","resnet50","vit_b_16"]
    protocols=["P1","P1-prime","P2","P3"]
    fig,axes=plt.subplots(2,3,figsize=(13,7.2),sharey=True)
    for ax,arch in zip(axes.ravel(),arch_order):
        g=acc[acc.architecture==arch].set_index("protocol").loc[protocols]
        ax.errorbar(protocols,100*g["mean"],yerr=100*g.sd,marker="o",lw=2,capsize=3,color=BLUE)
        ax.set_title(arch.replace("_"," "))
        ax.set_ylim(93,99)
        ax.set_ylabel("Accuracy (%)")
    fig.suptitle("SIPaKMeD classifier scores under matched split protocols (mean +/- SD, five seeds)",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,"classification_protocol_scores.png")

    c=pd.read_csv(ROOT/"results/classification_primary_contrasts.csv")
    c=c[c.left_protocol=="P1"].copy().sort_values("mean_difference")
    fig,ax=plt.subplots(figsize=(10.5,5.4))
    y=np.arange(len(c)); x=100*c.mean_difference
    # CIs are standardized-effect intervals, so show delta SD as horizontal uncertainty and effect text separately.
    ax.errorbar(x,y,xerr=100*c.sd_difference,fmt="o",color=BLUE,ecolor="#8FB5C4",capsize=4)
    ax.axvline(0,color=INK,lw=1)
    ax.set_yticks(y,[a.replace("_"," ") for a in c.architecture])
    ax.set_xlabel("Paired P1 - P2 accuracy (percentage points); error bar = SD across seeds")
    ax.set_title("All point estimates favor P1, but none survives the 12-test BH family",
                 loc="left",weight="bold")
    for yi,row in enumerate(c.itertuples()):
        ax.text(x.iloc[yi]+.12,yi,f"q={row.q_value:.3f}, d={row.effect_size:.2f}",va="center",fontsize=8)
    save(fig,"classification_delta_forest.png")

    runs=read_jsons("results/classification/*.json")
    piv=runs.pivot_table(index=["architecture","seed"],columns="protocol",values="accuracy")
    delta=(piv.P1-piv.P2).unstack("seed").loc[arch_order]*100
    fig,ax=plt.subplots(figsize=(10.5,5.2))
    im=ax.imshow(delta,cmap="RdBu_r",aspect="auto",vmin=-max(abs(delta.min().min()),abs(delta.max().max())),vmax=max(abs(delta.min().min()),abs(delta.max().max())))
    ax.set_xticks(range(len(delta.columns)),delta.columns)
    ax.set_yticks(range(len(delta.index)),[x.replace("_"," ") for x in delta.index])
    ax.set_xlabel("Seed"); ax.set_title("Seed-level P1 - P2 accuracy differences (percentage points)",loc="left",weight="bold")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]): ax.text(j,i,f"{delta.iloc[i,j]:+.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="percentage points")
    save(fig,"classification_seed_deltas.png")

    ordering=pd.read_csv(ROOT/"results/classification_protocol_ordering.csv")
    fig,ax=plt.subplots(figsize=(10.2,4.4))
    vals=ordering.set_index("architecture").loc[arch_order,["P1","P2","P3"]]*100
    im=ax.imshow(vals,cmap="YlGnBu",aspect="auto")
    ax.set_xticks(range(3),["P1","P2","P3"]); ax.set_yticks(range(len(vals)),[x.replace("_"," ") for x in vals.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]): ax.text(j,i,f"{vals.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8,color=INK)
    ax.set_title("The preregistered P3 <= P2 <= P1 model ordering fails for four of six classifiers",
                 loc="left",weight="bold")
    fig.colorbar(im,ax=ax,label="mean accuracy (%)")
    save(fig,"classification_ordering.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for protocol,color in [("P1",RED),("P2",BLUE),("P3",TEAL)]:
        subset=runs[(runs.architecture=="resnet50") & (runs.protocol==protocol)]
        a=np.array([[e["train_loss"] for e in h] for h in subset.history])
        axes[0].plot(range(1,a.shape[1]+1),a.mean(0),label=protocol,color=color)
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Mean training loss")
    axes[0].set_title("ResNet-50 training trajectory")
    runt=[]
    for _,r in runs.iterrows(): runt.append({"architecture":r.architecture,"minutes":r.elapsed_seconds/60})
    rdf=pd.DataFrame(runt)
    axes[1].boxplot([rdf[rdf.architecture==a].minutes for a in arch_order],tick_labels=[a.replace("_","\n") for a in arch_order],showfliers=False)
    axes[1].set_ylabel("Wall time per 20-epoch run (minutes)"); axes[1].set_title("Measured runtime on RTX A6000")
    axes[0].legend(frameon=False)
    fig.suptitle("Optimization and computation diagnostics are part of the evidence record",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig,"classification_training_runtime.png")


def cx22_label_figure():
    base=ROOT/"data/extracted/cx22/Cx22-Pair/Cx22-Pair"
    def first_masks(path,key):
        with h5py.File(path,"r") as h:
            ref=np.asarray(h[key]).ravel(order="F")[0]
            inner=h[ref]
            return [np.asarray(h[r],dtype=np.uint8).T for r in np.asarray(inner).ravel(order="F")]
    cyto=first_masks(base/"cyto/cyto_ins.mat","cyto_ins")
    nuc=first_masks(base/"nuc/nuc_ins.mat","nuc_ins")
    cu=np.any(cyto,axis=0); nu=np.any(nuc,axis=0)
    overlay=np.ones((512,512,3),dtype=float)
    overlay[cu]=mpl.colors.to_rgb("#6EC5D9")
    overlay[nu]=mpl.colors.to_rgb("#C4455A")
    inst=np.ones((512,512,3),dtype=float)
    for i,m in enumerate(cyto): inst[m.astype(bool)]=mpl.colors.to_rgb(COLORS[i%len(COLORS)])
    fig,axes=plt.subplots(1,4,figsize=(14,3.8))
    for ax,img,title,cmap in [
        (axes[0],cu,"Cytoplasm union","Blues"),(axes[1],nu,"Nucleus union","Reds"),
        (axes[2],overlay,"Two-channel semantic target",None),(axes[3],inst,f"Cytoplasm instances (n={len(cyto)})",None)]:
        ax.imshow(img,cmap=cmap); ax.set_title(title,fontsize=10,weight="bold"); ax.axis("off")
    fig.suptitle("Cx22 label structure decoded directly from the released MATLAB-v7.3 cells",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.text(.04,.01,"The corresponding CCEDD RGB source is not redistributed in this local report package; masks are shown exactly as released.",fontsize=9,color=MUTED)
    fig.tight_layout(rect=(0,.04,1,.92))
    save(fig,"cx22_label_structure.png")


def cx22_figures():
    s=pd.read_csv(ROOT/"results/cx22_segmentation_analysis/cx22_segmentation_summary.csv")
    primary=s[s.metric.isin(["macro_dice","instance_dice"])].copy()
    sem=primary[primary.task=="semantic_segmentation"]
    archs=["deeplabv3plus_resnet50","fpn_resnet34","manet_resnet34","unet_resnet34","unetplusplus_resnet34"]
    fig,axes=plt.subplots(2,3,figsize=(13,7.4),sharey=True)
    protocols=["P1","P1-prime","P2","P3","OFFICIAL"]
    for ax,arch in zip(axes.ravel(),archs):
        g=sem[sem.architecture==arch].set_index("protocol").reindex(protocols).dropna()
        ax.errorbar(g.index,100*g["mean"],yerr=100*g.sd,marker="o",capsize=3,color=BLUE)
        ax.set_title(arch.replace("_"," "),fontsize=10)
        ax.tick_params(axis="x",rotation=25); ax.set_ylim(80,92); ax.set_ylabel("Macro Dice (%)")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Cx22 semantic segmentation: pooled protocols and the descriptive official split",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,"cx22_semantic_scores.png")

    inst=primary[primary.task=="instance_segmentation"]
    fig,ax=plt.subplots(figsize=(10.5,5.2))
    xs=np.arange(5); w=.34
    for i,arch in enumerate(["maskrcnn_resnet50_fpn","maskrcnn_resnet50_fpn_v2"]):
        g=inst[inst.architecture==arch].set_index("protocol").reindex(protocols)
        ax.bar(xs+(i-.5)*w,100*g["mean"],w,yerr=100*g.sd,capsize=3,label=arch.replace("_"," "),color=[BLUE,TEAL][i])
    ax.set_xticks(xs,protocols); ax.set_ylim(88,94); ax.set_ylabel("Matched instance Dice (%)")
    ax.set_title("Cx22 instance segmentation",loc="left",weight="bold"); ax.legend(frameon=False)
    save(fig,"cx22_instance_scores.png")

    # Per-channel semantic heatmap for P1/P2/P3 and all architectures.
    sub=s[(s.task=="semantic_segmentation") & (s.protocol.isin(["P1","P2","P3"])) & (s.metric.isin(["cytoplasm_dice","nucleus_dice"]))]
    labels=[]; rows=[]
    for arch in archs:
        for protocol in ["P1","P2","P3"]:
            g=sub[(sub.architecture==arch)&(sub.protocol==protocol)].set_index("metric")
            labels.append(arch.replace("_"," ")+" / "+protocol)
            rows.append([100*g.loc["cytoplasm_dice","mean"],100*g.loc["nucleus_dice","mean"]])
    arr=np.array(rows)
    fig,ax=plt.subplots(figsize=(8.5,8.5))
    im=ax.imshow(arr,cmap="YlGnBu",aspect="auto",vmin=80,vmax=97)
    ax.set_xticks([0,1],["Cytoplasm Dice","Nucleus Dice"]); ax.set_yticks(range(len(labels)),labels,fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(2): ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=8,color=INK)
    ax.set_title("Nucleus segmentation is consistently the harder semantic channel",loc="left",weight="bold")
    fig.colorbar(im,ax=ax,label="Dice (%)")
    save(fig,"cx22_channel_heatmap.png")

    c=pd.read_csv(ROOT/"results/cx22_segmentation_analysis/cx22_segmentation_primary.csv")
    c=c[c.left_protocol=="P1"].copy()
    c["label"]=c.architecture.str.replace("_"," ")
    c["delta_pp"]=100*c.mean_difference
    c=c.sort_values(["task","delta_pp"])
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharex=True)
    for ax,(task,g) in zip(axes,c.groupby("task")):
        y=np.arange(len(g))
        ax.errorbar(g.delta_pp,y,xerr=100*g.sd_difference,fmt="o",color=BLUE,ecolor="#8FB5C4",capsize=3)
        ax.axvline(0,color=INK,lw=1)
        ax.set_yticks(y,g.label,fontsize=8); ax.set_xlabel("P1 - P2 (percentage points; error bar = SD)")
        ax.set_title(task.replace("_"," ").title())
        for yi,row in enumerate(g.itertuples()): ax.text(row.delta_pp+.04,yi,f"q={row.q_value:.3f}",va="center",fontsize=8)
    fig.suptitle("Cx22 P1-P2 effects are small; no confirmatory contrast survives BH correction",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig,"cx22_contrasts.png")

    runs=read_jsons("results/cx22_segmentation/*.json")
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    semrun=runs[(runs.task=="semantic_segmentation")&(runs.architecture=="unet_resnet34")&(runs.protocol=="P1")]
    for _,r in semrun.iterrows(): axes[0].plot([e["epoch"] for e in r.history],[e["macro_dice"]*100 for e in r.history],alpha=.7,label=f"seed {r.seed}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Macro Dice (%)"); axes[0].set_title("U-Net P1 test trajectory")
    rr=[]
    for _,r in runs.iterrows(): rr.append({"task":r.task,"hours":r.elapsed_seconds/3600})
    rdf=pd.DataFrame(rr)
    axes[1].boxplot([rdf[rdf.task==t].hours for t in ["semantic_segmentation","instance_segmentation"]],tick_labels=["semantic","instance"],showfliers=True)
    axes[1].set_ylabel("Wall time per 30-epoch cell (hours)"); axes[1].set_yscale("log"); axes[1].set_title("Runtime bottleneck was concentrated in instance models")
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Cx22 optimization and runtime evidence",x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig,"cx22_training_runtime.png")


def acceptance_figure():
    a=pd.read_csv(ROOT/"reports/generated/acceptance_contract.csv")
    counts=a.status.value_counts()
    colors={"PASS":TEAL,"FAIL":RED,"PENDING":GOLD}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1,2.2]})
    axes[0].pie([counts.get(k,0) for k in ["PASS","FAIL","PENDING"]],labels=["PASS","FAIL","PENDING"],colors=[colors[k] for k in ["PASS","FAIL","PENDING"]],autopct="%1.0f%%",startangle=90)
    axes[0].set_title("Contract status")
    short=[]
    for r in a.itertuples(): short.append(f"{r.section}: {r.criterion[:45]}")
    y=np.arange(len(a))
    axes[1].scatter([1]*len(a),y,s=75,c=[colors[s] for s in a.status])
    for yi,(label,status) in enumerate(zip(short,a.status)):
        axes[1].text(1.05,yi,label,va="center",fontsize=7.5,color=INK,clip_on=False)
        axes[1].text(.95,yi,status,va="center",ha="right",fontsize=7.3,weight="bold",color=colors[status])
    axes[1].set_xlim(.55,2.8); axes[1].set_ylim(-1,len(a)); axes[1].axis("off")
    fig.suptitle("Fail-closed verification: absence of evidence remains PENDING, not PASS",
                 x=.04,ha="left",fontsize=15,weight="bold",color=INK)
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig,"acceptance_dashboard.png")


def execution_figure():
    fig,ax=plt.subplots(figsize=(13,5.4)); ax.set_xlim(0,13); ax.set_ylim(0,5.5); ax.axis("off")
    stages=[(0.3,"Raw owner\narchives",BLUE),(2.35,"Checksum +\nextract",BLUE),(4.4,"Parent maps +\ngeometry gates",TEAL),(6.45,"Fixed split\nmanifests",TEAL),(8.5,"GPU run JSON\n+ histories",PURPLE),(10.55,"Summary +\nstatistics",GOLD)]
    for x,text_,color in stages: box(ax,(x,2.7),(1.6,1.05),text_,color,"white",10)
    for (x,_,_),(x2,_,_) in zip(stages[:-1],stages[1:]):  # noqa: RUF007
        arrow(ax,(x+1.6,3.22),(x2,3.22))
    box(ax,(4.4,.65),(3.65,1.0),"Fail closed if counts, bounds, hashes,\nseeds, runtime, or completeness disagree",RED,"#FCECEF",10)
    arrow(ax,(6.2,2.7),(6.2,1.65),color=RED)
    box(ax,(9.15,.65),(3.0,1.0),"Release artifacts\nCSV + Parquet + PDF",TEAL,"#E9F6F2",10)
    arrow(ax,(11.35,2.7),(10.7,1.65),color=TEAL)
    ax.text(.3,4.8,"Reproducibility architecture",fontsize=17,weight="bold",color=INK)
    ax.text(.3,4.35,"Every stochastic result retains its seed, split hash, input hash, Git SHA, runtime, hardware, and epoch history.",fontsize=10,color=MUTED)
    save(fig,"reproducibility_pipeline.png")


def latex_escape(value) -> str:
    s=str(value)
    if s.startswith("RAWLATEX:"):
        return s.removeprefix("RAWLATEX:")
    repl={"\\":"\\textbackslash{}","&":"\\&","%":"\\%","$":"\\$","#":"\\#","_":"\\_","{":"\\{","}":"\\}","~":"\\textasciitilde{}","^":"\\textasciicircum{}"}
    return "".join(repl.get(ch,ch) for ch in s)


def longtable(path: Path, columns: list[str], rows: list[list], caption: str, label: str, widths: list[str] | None=None):
    spec="".join(widths) if widths else "l"*len(columns)
    head=" & ".join(latex_escape(c) for c in columns)+" \\\\"
    body=[]
    for row in rows: body.append(" & ".join(latex_escape(v) for v in row)+" \\\\ ")
    text=(f"\\begin{{longtable}}{{{spec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
          f"\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"+"\n".join(body)+"\n\\end{longtable}\n")
    path.write_text(text)


def table_files():
    # Complete seed-level classifier ledger.
    runs=read_jsons("results/classification/*.json").sort_values(["architecture","protocol","seed"])
    rows=[]
    for r in runs.itertuples():
        rows.append([r.architecture,r.protocol,r.seed,r.train_crops,r.test_crops,f"{100*r.accuracy:.3f}",f"{100*r.macro_f1:.3f}",f"{100*r.balanced_accuracy:.3f}",f"{r.elapsed_seconds/60:.2f}",r.gpu_visible_devices])
    longtable(TABLES/"classification_runs.tex",["Architecture","Protocol","Seed","Train","Test","Acc. (percent)","Macro-F1 (percent)","Bal. acc. (percent)","min","GPU"],rows,
              "Complete SIPaKMeD classification run ledger. All values are final-epoch results.","tab:all-class-runs",["p{2.25cm}","p{1.1cm}","r","r","r","r","r","r","r","r"])

    probe=read_jsons("results/probe/*.json").sort_values(["image_mode","shuffled_labels","seed"])
    rows=[]
    for r in probe.itertuples():
        rows.append([r.image_mode,"yes" if r.shuffled_labels else "no",r.seed,r.n_parents,r.train_crops,r.test_crops,f"{100*r.chance_accuracy:.4f}",f"{100*r.test_accuracy:.3f}",f"{r.elapsed_seconds/60:.2f}",r.gpu_visible_devices])
    longtable(TABLES/"probe_runs.tex",["Mode","Shuffle","Seed","Parents","Train","Test","Chance (percent)","Accuracy (percent)","min","GPU"],rows,
              "Complete parent-identity probe ledger.","tab:all-probe-runs",["p{2.4cm}","p{1.0cm}","r","r","r","r","r","r","r","r"])

    cx=read_jsons("results/cx22_segmentation/*.json").sort_values(["task","architecture","protocol","seed"])
    rows=[]
    for r in cx.itertuples():
        metric=r.macro_dice if r.task=="semantic_segmentation" else r.instance_dice
        secondary=r.macro_iou if r.task=="semantic_segmentation" else r.instance_iou
        extra=r.nucleus_dice if r.task=="semantic_segmentation" else r.recall70
        rows.append([r.task.replace("_segmentation",""),f"RAWLATEX:\\path{{{r.architecture}}}",r.protocol,r.seed,r.train_crops,r.test_crops,f"{100*metric:.3f}",f"{100*secondary:.3f}",f"{100*extra:.3f}",f"{r.elapsed_seconds/3600:.2f}"])
    longtable(TABLES/"cx22_runs.tex",["Task","Architecture","Protocol","Seed","Train","Test","Primary (percent)","IoU (percent)","Nuc./R70 (percent)","hours"],rows,
              "Complete Cx22 run ledger. Primary is macro Dice for semantic models and matched-instance Dice for instance models; the ninth column is nucleus Dice or recall at Dice 0.70.","tab:all-cx22-runs",["p{1.6cm}","p{4.4cm}","p{1.5cm}","r","r","r","r","r","r","r"])

    sources=json.loads((ROOT/"data_sources.json").read_text())["sources"]
    rows=[[x["dataset"],x["filename"],f"{x['size']:,}",f"RAWLATEX:\\seqsplit{{{x['sha256']}}}",x["terms"]] for x in sources]
    longtable(TABLES/"data_sources.tex",["Dataset","File","Bytes","SHA-256","Terms note"],rows,
              "Declared owner-source files, exact byte sizes, checksums, and use notes.","tab:data-sources",["p{1.5cm}","p{3.5cm}","r","p{8.0cm}","p{9.0cm}"])

    # Git chronology records decisions with short hashes and dates.
    output=subprocess.check_output(["git","log","--reverse","--date=short","--format=%h|%ad|%s","--all"],cwd=ROOT,text=True)
    rows=[line.split("|",2) for line in output.splitlines()]
    longtable(TABLES/"git_timeline.tex",["Commit","Date","Recorded change"],rows,
              "Complete repository decision chronology through the frozen result set.","tab:git-timeline",["p{1.5cm}","p{2.0cm}","p{11.5cm}"])

    # Split manifest hashes.
    man=pd.read_csv(ROOT/"splits/generated/sipakmed_manifest.csv")
    rows=[[r.protocol,r.seed,r.path,f"RAWLATEX:\\seqsplit{{{r.sha256}}}"] for r in man.itertuples()]
    longtable(TABLES/"split_hashes.tex",["Protocol","Seed","File","SHA-256"],rows,
              "Released SIPaKMeD split manifests and hashes.","tab:split-hashes",["p{1.5cm}","r","p{4.8cm}","p{6.4cm}"])

    structural=pd.read_csv(ROOT/"results/table1_seed_level.csv")
    rows=[]
    for r in structural.itertuples():
        rows.append([r.dataset,r.seed,f"{100*r.p1_straddle_fraction:.3f}",f"{100*r.p1_leaked_test_crop_fraction:.3f}",f"{100*r.p2_straddle_fraction:.3f}",f"{100*r.p3_straddle_fraction:.3f}"])
    longtable(TABLES/"structural_seed_runs.tex",["Dataset","Seed","P1 parent straddle (percent)","P1 test parent in train (percent)","P2 straddle (percent)","P3 straddle (percent)"],rows,
              "Complete structural seed ledger.","tab:structural-seed-ledger",["p{2.0cm}","p{1.0cm}","p{2.5cm}","p{2.8cm}","p{2.5cm}","p{2.5cm}"])

    pixel=pd.read_csv(ROOT/"results/pixel_overlap_seed_level.csv")
    rows=[]
    for r in pixel.itertuples():
        rows.append([r.protocol,r.seed,r.cross_split_pairs,r.overlapping_cross_split_pairs,f"{100*r.overlapping_cross_split_pair_fraction:.3f}",r.test_crops_with_train_pixel_overlap,f"{100*r.test_crop_pixel_overlap_fraction:.3f}"])
    longtable(TABLES/"pixel_overlap_runs.tex",["Protocol","Seed","Cross-split pairs","Overlapping pairs","Pair overlap (percent)","Test crops overlapping train","Test-crop overlap (percent)"],rows,
              "Complete SIPaKMeD literal source-pixel overlap ledger.","tab:pixel-seed-ledger",["p{1.4cm}","p{1.0cm}","p{2.0cm}","p{2.0cm}","p{2.0cm}","p{2.8cm}","p{2.3cm}"])

    def contrast_rows(primary_path, secondary_path):
        primary=pd.read_csv(primary_path).assign(family="primary")
        secondary=pd.read_csv(secondary_path).assign(family="secondary",q_value=np.nan)
        frame=pd.concat([primary,secondary],ignore_index=True,sort=False)
        out=[]
        for r in frame.itertuples():
            q="--" if pd.isna(r.q_value) else f"{r.q_value:.4f}"
            out.append([r.family,f"RAWLATEX:\\path{{{r.architecture}}}",r.left_protocol,r.right_protocol,r.n,f"{100*r.mean_difference:.3f}",f"{100*r.sd_difference:.3f}",r.test,f"{r.p_value:.4f}",q,r.effect_name,f"{r.effect_size:.3f}",f"[{r.effect_ci_low:.3f}, {r.effect_ci_high:.3f}]",f"{r.shapiro_p:.4f}"])
        return out

    rows=contrast_rows(ROOT/"results/classification_primary_contrasts.csv",ROOT/"results/classification_secondary_contrasts.csv")
    longtable(TABLES/"classification_contrasts.tex",["Family","Architecture","Left","Right","n","Mean diff. (pp)","SD (pp)","Test","p","BH q","Effect","Value","Effect 95 percent CI","Shapiro p"],rows,
              "Complete SIPaKMeD classifier contrast ledger. Secondary tests have no confirmatory BH q-value.","tab:all-class-contrasts",["p{1.4cm}","p{3.1cm}","p{1.0cm}","p{1.0cm}","r","r","r","p{1.7cm}","r","r","p{2.0cm}","r","p{2.6cm}","r"])

    rows=contrast_rows(ROOT/"results/cx22_segmentation_analysis/cx22_segmentation_primary.csv",ROOT/"results/cx22_segmentation_analysis/cx22_segmentation_secondary.csv")
    longtable(TABLES/"cx22_contrasts.tex",["Family","Architecture","Left","Right","n","Mean diff. (pp)","SD (pp)","Test","p","BH q","Effect","Value","Effect 95 percent CI","Shapiro p"],rows,
              "Complete Cx22 contrast ledger. Semantic and instance primary families were corrected separately by task.","tab:all-cx22-contrasts",["p{1.4cm}","p{4.1cm}","p{1.0cm}","p{1.0cm}","r","r","r","p{1.7cm}","r","r","p{2.0cm}","r","p{2.6cm}","r"])

    class_rank=pd.read_csv(ROOT/"results/classification_ranking_stability.csv").assign(task="classification")
    cx_rank=pd.read_csv(ROOT/"results/cx22_segmentation_analysis/cx22_segmentation_ranking.csv")
    rank=pd.concat([class_rank,cx_rank],ignore_index=True,sort=False)
    rows=[]
    for r in rank.itertuples():
        rows.append([r.task.replace("_", " "),r.n_architectures,r.n_seeds,f"{r.effect_size:.3f}",f"[{r.effect_ci_low:.3f}, {r.effect_ci_high:.3f}]",f"{r.p_value:.4f}",f"{r.reordered_pairs}/{r.total_pairs}",r.tie_changes])
    longtable(TABLES/"ranking_results.tex",["Task","Arch.","Seeds","Kendall tau","95 percent CI","p","Reordered pairs","Tie changes"],rows,
              "Complete secondary P1-versus-P2 rank-stability ledger.","tab:all-rank-results",["p{3.2cm}","p{1.3cm}","p{1.0cm}","p{1.7cm}","p{2.3cm}","p{1.2cm}","p{1.8cm}","p{1.4cm}"])

    invalid=[]
    attempt_groups=[("split-library failure",ROOT/"results/attempts/classification_sklearn_1.7.2"),("smoke test",ROOT/"results/attempts/smoke_base_env")]
    for reason,directory in attempt_groups:
        for path in sorted(directory.glob("*.json")):
            record=json.loads(path.read_text())
            name=record.get("architecture",record.get("image_mode",""))
            if record.get("task")=="parent_identity_probe":
                name=f"{name}/{record.get('image_mode','')}"
            metric=record.get("accuracy",record.get("test_accuracy",np.nan))
            invalid.append([reason,name,record.get("protocol","--") or "--",record.get("seed","--"),record.get("epochs","--"),record.get("train_crops","--"),record.get("test_crops","--"),f"{100*metric:.3f}" if pd.notna(metric) else "--","No"])
    longtable(TABLES/"invalid_attempts.tex",["Reason","Architecture/mode","Protocol","Seed","Epochs","Train","Test","Recorded metric (percent)","Eligible"],invalid,
              "Complete excluded-attempt ledger. Metrics are shown only to demonstrate that exclusion was rule-based, not score-based.","tab:invalid-attempts",["p{3.7cm}","p{3.3cm}","p{1.5cm}","r","r","r","r","r","p{1.2cm}"])


def main():
    lifecycle_figure(); hierarchy_figure(); dataset_figures(); provenance_figures(); split_figure()
    structural_figures(); overlap_and_duplicate_figures(); transform_plate(); probe_figures()
    classification_figures(); cx22_label_figure(); cx22_figures(); acceptance_figure(); execution_figure()
    table_files()
    print(f"generated assets in {OUT}")


if __name__ == "__main__":
    main()
