#!/usr/bin/env python3
"""Draw every new graph of the lifecycle record from committed outputs.

Style follows ``scripts/study2/figstyle.py``: the validated categorical order
(blue, orange, aqua, yellow), solid hairline grids, ink-coloured text, legends for
two or more series, and no dual axes. Protocol identity is fixed throughout:
P1 (crop-level) orange, P2 (parent-level) blue, P3 (component-level) aqua.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "study2"))
sys.path.insert(0, str(ROOT / "src"))
import figstyle as style  # noqa: E402

OUT = ROOT / "record" / "figures"
A = ROOT / "results" / "study2" / "analysis"
F = ROOT / "results" / "study2" / "followup"
R = ROOT / "results" / "study2" / "run_records"
IST = timezone(timedelta(hours=5, minutes=30))
NAMES = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNet-V2", "vit_b_16": "ViT-B/16",
    "convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T", "regnet_y_1_6gf": "RegNetY-1.6GF",
    "efficientnet_v2_s": "EfficientNetV2-S", "maxvit_t": "MaxViT-T", "dinov2_vits14": "DINOv2 ViT-S/14",
}
ORDER = list(NAMES)
PROTOCOL_COLOR = {"P1": style.ORANGE, "P2": style.BLUE, "P3": style.AQUA, "P1-prime": style.GRAY_SERIES}


def save(figure, name: str) -> None:
    figure.savefig(OUT / f"{name}.pdf")
    figure.savefig(OUT / f"{name}.png")
    plt.close(figure)
    print(name)


def t_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = values.mean()
    if len(values) < 2:
        return mean, mean, mean
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def ist(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(IST).dt.tz_localize(None)


def title(ax, text):
    ax.set_title(text, loc="left", color=style.INK)


# ---------------------------------------------------------------- timelines
def fig_timeline() -> None:
    log = subprocess.check_output(["git", "-C", str(ROOT), "log", "--all", "--format=%ad|%s",
                                   "--date=format:%Y-%m-%d %H:%M"], text=True).strip().splitlines()
    commits = pd.DataFrame([line.split("|", 1) for line in log], columns=["date", "message"])
    commits["date"] = pd.to_datetime(commits.date)

    def phase(row):
        if row.date < pd.Timestamp("2026-10-01"):
            return "Study 1"
        if row.date < pd.Timestamp("2026-10-19 13:00"):
            return "Study 2"
        return "Amendment 8"
    commits["phase"] = commits.apply(phase, axis=1)
    colors = {"Study 1": style.BLUE, "Study 2": style.ORANGE, "Amendment 8": style.AQUA}
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [1, 1.6]})
    ax = axes[0]
    daily = commits.groupby([commits.date.dt.floor("D"), "phase"]).size().reset_index(name="n")
    for name in ("Study 1", "Study 2", "Amendment 8"):
        group = daily.loc[daily.phase == name]
        ax.bar(group.date, group.n, width=0.8, color=colors[name], label=name)
    for date, text, height in ((pd.Timestamp("2026-08-22"), "Study 1 registered and built", 55),
                               (pd.Timestamp("2026-09-07"), "Study 1 evidence frozen", 30),
                               (pd.Timestamp("2026-09-13"), "Study 1 record", 18),
                               (pd.Timestamp("2026-10-18"), "Study 2 and Amendment 8", 30)):
        ax.annotate(text, (date, height), xytext=(5, 0), textcoords="offset points", ha="left" if date.month != 10 else "right",
                    va="center", fontsize=6.5, color=style.INK_2)
        ax.plot([date, date], [0, height], color=style.MUTED, linewidth=0.6, zorder=0)
    ax.set_ylabel("Commits per day")
    ax.set_ylim(0, 62)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.62, 1.0))
    title(ax, "Whole project, August to October 2026")

    ax = axes[1]
    runs = pd.read_csv(R / "training_runs.csv")
    evals = pd.read_csv(R / "evaluation_runs.csv")
    start, end = ist(runs.started_at_utc), ist(runs.completed_at_utc)
    s2 = runs.experiment.isin(["phase1", "factorial", "grl", "cric"])
    e_end = ist(evals.completed_at_utc)
    e_start = e_end - pd.to_timedelta(evals.elapsed_seconds, unit="s")
    spans = [
        ("Study 2 implementation and dry runs", pd.Timestamp("2026-10-18 00:30"), pd.Timestamp("2026-10-18 05:07"), style.MUTED),
        ("Study 2 matrix (1,341 runs)", start[s2].min(), end[s2].max(), style.ORANGE),
        ("Study 2 registered analysis", pd.Timestamp("2026-10-19 12:56"), pd.Timestamp("2026-10-19 13:07"), style.INK_2),
        ("Amendment 8 implementation", pd.Timestamp("2026-10-19 13:15"), pd.Timestamp("2026-10-19 14:02"), style.MUTED),
        ("Follow-up evaluation arms (640)", e_start.min(), e_end.max(), style.GRAY_SERIES),
        ("Follow-up training arms (940)", start[~s2].min(), end[~s2].max(), style.AQUA),
        ("Follow-up registered analysis", pd.Timestamp("2026-10-20 02:57"), pd.Timestamp("2026-10-20 03:20"), style.INK_2),
    ]
    for position, (label, a, b, color) in enumerate(spans):
        ax.barh(position, (b - a) / pd.Timedelta(days=1), left=mdates.date2num(a), height=0.55, color=color)
        ax.annotate(label, (mdates.date2num(b), position), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6.8, color=style.INK)
    for date, label, ha in ((pd.Timestamp("2026-10-18 05:07"), "Study 2 registered", "right"),
                            (pd.Timestamp("2026-10-18 06:29"), "Amend. 7", "left"),
                            (pd.Timestamp("2026-10-19 14:02"), "Amend. 8", "center"),
                            (pd.Timestamp("2026-10-20 01:25"), "Amend. 9", "center")):
        ax.axvline(mdates.date2num(date), color=style.BASELINE, linewidth=0.8, zorder=0)
        ax.annotate(label, (mdates.date2num(date), -0.9), xytext=(-2 if ha == "right" else (2 if ha == "left" else 0), 0),
                    textcoords="offset points", ha=ha, va="top", fontsize=6.3, color=style.INK_2)
    ax.set_yticks([])
    ax.set_ylim(len(spans) - 0.4, -1.8)
    ax.set_xlim(mdates.date2num(pd.Timestamp("2026-10-18 00:00")), mdates.date2num(pd.Timestamp("2026-10-20 12:00")))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(axis="y", visible=False)
    title(ax, "Study 2 and Amendment 8, hour by hour (IST)")
    figure.tight_layout(h_pad=1.2)
    save(figure, "timeline")


def fig_gpu_schedule() -> None:
    runs = pd.read_csv(R / "training_runs.csv")
    evals = pd.read_csv(R / "evaluation_runs.csv")
    gpu_of = {}
    for line in (R / "controller_followup.log").read_text().splitlines():
        if line.startswith("started "):
            parts = line.split()
            gpu_of[parts[1].split(":", 1)[1]] = int(parts[2].split("=")[1])
    evals["gpu"] = evals.run.map(gpu_of)
    evals["end"] = ist(evals.completed_at_utc)
    evals["start"] = evals.end - pd.to_timedelta(evals.elapsed_seconds, unit="s")
    runs["start"], runs["end"] = ist(runs.started_at_utc), ist(runs.completed_at_utc)
    runs["gpu"] = runs.gpu_visible_devices.astype(int)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.2))
    panels = [
        (axes[0], runs.loc[runs.experiment.isin(["phase1", "factorial", "grl", "cric"])],
         {"phase1": ("Phase 1", style.BLUE), "factorial": ("2C/2B", style.ORANGE), "grl": ("2A GRL", style.AQUA),
          "cric": ("CRIC", style.YELLOW)}, "Study 2 matrix"),
        (axes[1], pd.concat([runs.loc[runs.experiment.isin(["carrier", "linadv", "normalization"])],
                             evals.assign(experiment=evals.experiment.replace({"factorial_eval": "eval", "leace": "eval"}))]),
         {"eval": ("evaluation only", style.GRAY_SERIES), "carrier": ("carrier factorial", style.ORANGE),
          "linadv": ("linear adversary", style.AQUA), "normalization": ("normalisation", style.BLUE)},
         "Amendment 8 matrix"),
    ]
    for ax, frame, colors, name in panels:
        gpus = sorted(frame.gpu.dropna().unique())
        row = {g: i for i, g in enumerate(gpus)}
        for experiment, (label, color) in colors.items():
            group = frame.loc[frame.experiment == experiment]
            ax.barh(group.gpu.map(row), (group.end - group.start) / pd.Timedelta(days=1), left=mdates.date2num(group.start),
                    height=0.7, color=color, linewidth=0, label=label)
        ax.set_yticks(range(len(gpus)), [f"GPU {int(g)}" for g in gpus])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=7)
        title(ax, f"{name}: every run on its GPU (IST)")
    figure.tight_layout(h_pad=1.6)
    save(figure, "gpu_schedule")


def fig_runtime() -> None:
    runs = pd.read_csv(R / "training_runs.csv")
    evals = pd.read_csv(R / "evaluation_runs.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    phase1 = runs.loc[runs.experiment == "phase1"]
    order = phase1.groupby("architecture").elapsed_seconds.median().sort_values().index
    for position, arch in enumerate(order):
        values = phase1.loc[phase1.architecture == arch, "elapsed_seconds"] / 60
        jitter = np.random.default_rng(position).uniform(-0.18, 0.18, len(values))
        ax.scatter(values, position + jitter, s=5, color=style.BLUE, alpha=0.35, linewidths=0)
        style.ring(ax, [values.median()], [position], style.BLUE, size=26)
    ax.set_yticks(range(len(order)), [NAMES[a] for a in order])
    ax.set_xlabel("Minutes per 20-epoch run (Phase 1)")
    ax.grid(axis="y", visible=False)
    title(ax, "Run time by architecture")
    ax = axes[1]
    hours = (runs.groupby("experiment").elapsed_seconds.sum() / 3600).to_dict()
    hours.update((evals.groupby("experiment").elapsed_seconds.sum() / 3600).to_dict())
    labels = {"phase1": "Phase 1", "factorial": "2C/2B", "grl": "2A GRL", "cric": "CRIC", "factorial_eval": "A8 ablations",
              "leace": "A8 LEACE", "carrier": "A8 carrier", "linadv": "A8 linear adv.", "normalization": "A8 normalisation"}
    items = [(labels[k], hours[k], style.ORANGE if k in ("phase1", "factorial", "grl", "cric") else style.AQUA)
             for k in labels]
    for position, (label, value, color) in enumerate(items):
        ax.barh(position, value, color=color, height=0.6)
        ax.annotate(f"{value:.1f}", (value, position), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=6.8, color=style.INK_2)
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.invert_yaxis()
    ax.set_xlabel("GPU-hours")
    ax.grid(axis="y", visible=False)
    title(ax, "Compute by arm")
    figure.tight_layout(w_pad=1.5)
    save(figure, "runtime")


# ---------------------------------------------------------------- Phase 1
def phase1_histories() -> pd.DataFrame:
    hist = pd.read_csv(R / "histories.csv")
    runs = pd.read_csv(R / "training_runs.csv")[["experiment", "run", "architecture", "protocol", "seed"]]
    return hist.merge(runs, on=["experiment", "run"])


def fig_training_curves(hist: pd.DataFrame) -> None:
    phase1 = hist.loc[hist.experiment == "phase1"]
    for metric, name, ylabel, log in (("accuracy", "training_curves", "Test accuracy (%)", False),
                                      ("train_loss", "loss_curves", "Training loss", True)):
        figure, axes = plt.subplots(3, 4, figsize=(7.2, 5.4), sharex=True)
        for ax, arch in zip(axes.flat, ORDER, strict=True):
            for protocol in ("P1", "P2", "P3"):
                group = phase1.loc[(phase1.architecture == arch) & (phase1.protocol == protocol)]
                means = group.groupby("epoch")[metric].mean() * (100 if metric == "accuracy" else 1)
                ax.plot(means.index, means.values, color=PROTOCOL_COLOR[protocol], linewidth=1.1, label=protocol)
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_minor_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            elif metric == "accuracy":
                lowest = phase1.loc[phase1.architecture == arch].groupby(["protocol", "epoch"]).accuracy.mean().min() * 100
                ax.set_ylim(min(88, np.floor(lowest)), 99)
            ax.set_title(NAMES[arch], fontsize=7.5, loc="left", color=style.INK)
            ax.tick_params(labelsize=6.5)
        for ax in axes[-1]:
            ax.set_xlabel("Epoch", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=7)
        axes[0, 0].legend(fontsize=6.5, loc="lower right" if metric == "accuracy" else "upper right")
        figure.tight_layout(h_pad=0.8, w_pad=0.6)
        save(figure, name)


def fig_phase1_points() -> None:
    phase1 = pd.read_csv(A / "runs" / "phase1_runs.csv")
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    order = phase1.loc[phase1.protocol == "P2"].groupby("architecture").accuracy.mean().sort_values().index
    for position, arch in enumerate(order):
        for shift, protocol in ((-0.22, "P1"), (0.0, "P2"), (0.22, "P3")):
            values = 100 * phase1.loc[(phase1.architecture == arch) & (phase1.protocol == protocol), "accuracy"]
            ax.scatter(values, np.full(len(values), position + shift), s=5, color=PROTOCOL_COLOR[protocol], alpha=0.35,
                       linewidths=0)
            style.ring(ax, [values.mean()], [position + shift], PROTOCOL_COLOR[protocol], size=22,
                       label=protocol if position == 0 else None)
    ax.set_yticks(range(len(order)), [NAMES[a] for a in order])
    ax.set_xlabel("Test accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    title(ax, "Phase 1: every run (small) and the mean over 20 seeds (ringed)")
    figure.tight_layout()
    save(figure, "phase1_points")


def diverging_cmap():
    return LinearSegmentedColormap.from_list("div", [style.BLUE, "#e9e8e3", style.ORANGE])


def fig_seed_heatmap() -> None:
    phase1 = pd.read_csv(A / "runs" / "phase1_runs.csv")
    wide = phase1.pivot_table(index=["architecture", "seed"], columns="protocol", values="accuracy")
    diff = (100 * (wide.P1 - wide.P2)).unstack("seed").reindex(ORDER)
    figure, ax = plt.subplots(figsize=(7.2, 3.6))
    limit = np.nanmax(np.abs(diff.values))
    image = ax.imshow(diff.values, cmap=diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(diff.index)), [NAMES[a] for a in diff.index])
    ax.set_xticks(range(len(diff.columns)), diff.columns, rotation=90, fontsize=6.5)
    ax.set_xlabel("Seed")
    ax.grid(False)
    bar = figure.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label("P1 minus P2 accuracy (pp)")
    positive = (diff.values > 0).sum()
    title(ax, f"Seed-level P1 minus P2: {positive} of {diff.size} pairs positive")
    figure.tight_layout()
    save(figure, "seed_heatmap")


def fig_calibration() -> None:
    phase1 = pd.read_csv(A / "runs" / "phase1_runs.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    order = ORDER[::-1]
    for ax, metric, label, scale in ((axes[0], "ece_15", "ECE, 15 bins (%)", 100), (axes[1], "nll", "NLL (nats)", 1)):
        for position, arch in enumerate(order):
            for shift, protocol in ((-0.2, "P1"), (0.0, "P2"), (0.2, "P3")):
                values = scale * phase1.loc[(phase1.architecture == arch) & (phase1.protocol == protocol), metric]
                mean, low, high = t_interval(values)
                ax.plot([low, high], [position + shift] * 2, color=PROTOCOL_COLOR[protocol], linewidth=1)
                style.ring(ax, [mean], [position + shift], PROTOCOL_COLOR[protocol], size=14,
                           label=protocol if position == 0 and ax is axes[0] else None)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [NAMES[a] for a in order])
    axes[0].legend(loc="upper right", fontsize=7)
    title(axes[0], "Calibration error")
    title(axes[1], "Negative log-likelihood")
    figure.tight_layout(w_pad=1.0)
    save(figure, "calibration")


def fig_confusion() -> None:
    matrices = json.loads((R / "confusion_matrices.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    short = {"Dyskeratotic": "Dysk.", "Koilocytotic": "Koil.", "Metaplastic": "Metap.", "Parabasal": "Parab.",
             "Superficial-Intermediate": "Sup.-Int.", "Negative for intraepithelial lesion": "NILM"}
    for row, (experiment, dataset) in enumerate((("phase1", "SIPaKMeD"), ("cric", "CRIC"))):
        for column, protocol in enumerate(("p1", "p2")):
            keys = [k for k in matrices if k.startswith(experiment + "/") and f"_{protocol}_seed" in k]
            total = np.sum([np.asarray(matrices[k]["matrix"]) for k in keys], axis=0)
            classes = [short.get(c, c) for c in matrices[keys[0]]["classes"]]
            normalised = total / total.sum(axis=1, keepdims=True)
            ax = axes[row, column]
            ax.imshow(normalised, cmap=LinearSegmentedColormap.from_list("seq", ["#ffffff", style.BLUE]), vmin=0, vmax=1)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    value = 100 * normalised[i, j]
                    ax.text(j, i, f"{value:.0f}" if value >= 0.5 else "", ha="center", va="center", fontsize=6.5,
                            color="white" if value > 60 else style.INK)
            ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=6.5)
            ax.set_yticks(range(len(classes)), classes, fontsize=6.5)
            ax.grid(False)
            ax.set_xlabel("Predicted", fontsize=7)
            if column == 0:
                ax.set_ylabel("True", fontsize=7)
            title(ax, f"{dataset}, {protocol.upper()}: row-normalised % ({len(keys)} runs pooled)")
    figure.tight_layout(h_pad=1.0, w_pad=1.0)
    save(figure, "confusion")


def fig_cric_recall() -> None:
    matrices = json.loads((R / "confusion_matrices.json").read_text())
    figure, ax = plt.subplots(figsize=(7.2, 2.8))
    recalls, support = {}, None
    for protocol in ("p1", "p2"):
        keys = [k for k in matrices if k.startswith("cric/") and f"_{protocol}_seed" in k]
        total = np.sum([np.asarray(matrices[k]["matrix"]) for k in keys], axis=0)
        recalls[protocol] = 100 * np.diag(total) / total.sum(axis=1)
        support = total.sum(axis=1) / len(keys)
        classes = matrices[keys[0]]["classes"]
    order = np.argsort(recalls["p2"])
    for position, index in enumerate(order):
        ax.plot([recalls["p2"][index], recalls["p1"][index]], [position, position], color=style.GRAY_SERIES, linewidth=1)
        style.ring(ax, [recalls["p1"][index]], [position], style.ORANGE, size=24, label="P1" if position == 0 else None)
        style.ring(ax, [recalls["p2"][index]], [position], style.BLUE, size=24, label="P2" if position == 0 else None)
    ax.set_yticks(range(len(order)), [f"{classes[i]} ({support[i]:.0f}/run)" for i in order], fontsize=7)
    ax.set_xlabel("Per-class recall, pooled over runs (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    title(ax, "CRIC: which classes lose most when slides are held out")
    figure.tight_layout()
    save(figure, "cric_recall")


def fig_cross_study() -> None:
    s1 = pd.read_csv(ROOT / "results" / "classification_primary_contrasts.csv")
    s1 = s1.loc[(s1.dataset == "SIPaKMeD") & (s1.left_protocol == "P1") & (s1.right_protocol == "P2")]
    h1 = json.loads((A / "h1_accuracy.json").read_text())
    per = {r["architecture"]: r for r in h1["per_architecture_descriptive"]}
    figure, ax = plt.subplots(figsize=(7.2, 2.9))
    archs = [a for a in ORDER if a in set(s1.architecture)]
    for position, arch in enumerate(archs):
        row = s1.loc[s1.architecture == arch].iloc[0]
        half = stats.t.ppf(0.975, row.n - 1) * row.sd_difference / np.sqrt(row.n)
        mean = 100 * row.mean_difference
        ax.plot([mean - 100 * half, mean + 100 * half], [position - 0.15] * 2, color=style.GRAY_SERIES, linewidth=1.2)
        style.ring(ax, [mean], [position - 0.15], style.GRAY_SERIES, size=22, label="Study 1 (5 seeds)" if position == 0 else None)
        r = per[arch]
        ax.plot([r["ci_low"], r["ci_high"]], [position + 0.15] * 2, color=style.ORANGE, linewidth=1.2)
        style.ring(ax, [r["mean_difference"]], [position + 0.15], style.ORANGE, size=22,
                   label="Study 2 (20 new seeds)" if position == 0 else None)
    style.zero_line(ax)
    ax.set_yticks(range(len(archs)), [NAMES[a] for a in archs])
    ax.set_xlabel("P1 minus P2 accuracy (pp), mean and 95% t interval")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    title(ax, "The six Study 1 architectures, re-run with fresh seeds")
    figure.tight_layout()
    save(figure, "cross_study")


# ---------------------------------------------------------------- mechanism arms (Study 2)
def fig_factorial_cells() -> None:
    fact = pd.read_csv(A / "runs" / "factorial_runs.csv")
    figure, ax = plt.subplots(figsize=(7.2, 2.6))
    styles = {"leak_high": (style.ORANGE, "o"), "leak_low": (style.ORANGE, "s"),
              "clean_high": (style.BLUE, "o"), "clean_low": (style.BLUE, "s")}
    archs = [a for a in ORDER if a in set(fact.architecture)]
    for position, arch in enumerate(archs):
        for shift, cell in zip((-0.24, -0.08, 0.08, 0.24), styles, strict=True):
            mean, low, high = t_interval(100 * fact.loc[(fact.architecture == arch) & (fact.protocol == cell), "accuracy"])
            color, marker = styles[cell]
            ax.plot([low, high], [position + shift] * 2, color=color, linewidth=1)
            style.ring(ax, [mean], [position + shift], color, size=20, marker=marker,
                       label=cell.replace("_", ", ") if position == 0 else None)
    ax.set_yticks(range(len(archs)), [NAMES[a] for a in archs])
    ax.set_xlabel("Accuracy on the common test set (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    title(ax, "2C cell means (10 seeds): leak above clean at both diversity levels")
    figure.tight_layout()
    save(figure, "factorial_cells")


def fig_transplant_arch() -> None:
    table = pd.read_csv(A / "mechanism" / "transplant_runs.csv")
    table = table.loc[table.subset == "all"]
    columns = [("acc_original", "orig."), ("acc_self", "self"), ("acc_sibling", "sibling"),
               ("acc_unseen_same_class", "unseen\nsame"), ("acc_unseen_other_class", "unseen\nother")]
    archs = [a for a in ORDER if a in set(table.architecture)]
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    for ax, arch in zip(axes, archs, strict=True):
        for leak, color, label in ((False, style.BLUE, "clean"), (True, style.ORANGE, "leak")):
            group = table.loc[(table.architecture == arch) & (table.cell.str.startswith("leak") == leak)]
            means = [100 * group[c].mean() for c, _ in columns]
            ax.plot(range(len(columns)), means, color=color, linewidth=1.1, label=label)
            style.ring(ax, range(len(columns)), means, color, size=12)
        ax.set_xticks(range(len(columns)), [c[1] for c in columns], fontsize=6)
        ax.grid(axis="x", visible=False)
        ax.set_title(NAMES[arch], fontsize=7.5, loc="left", color=style.INK)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=6.5, loc="lower left")
    figure.tight_layout(w_pad=0.6)
    save(figure, "transplant_arch")


def fig_grl_full() -> None:
    grl = pd.read_csv(A / "mechanism" / "grl_recoverability.csv")
    hist = pd.read_csv(R / "histories.csv")
    final = hist.loc[(hist.experiment == "grl") & (hist.epoch == 20), ["run", "adversary_within_class_accuracy"]]
    grl["run"] = grl.run_json.str.rsplit("/", n=1).str[1].str.replace(".json", "", regex=False)
    grl = grl.merge(final, on="run", how="left")
    lambdas = sorted(grl.adversary_lambda.unique())
    pos = {v: i for i, v in enumerate(lambdas)}
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for row, arch in enumerate(("resnet50", "convnext_tiny")):
        group = grl.loc[grl.architecture == arch]
        for protocol in ("P1", "P2"):
            sub = group.loc[group.protocol == protocol]
            for column, (metric, scale) in enumerate((("accuracy", 100), ("within_class_accuracy", 100),
                                                      ("adversary_within_class_accuracy", 100))):
                means = sub.groupby("adversary_lambda")[metric].mean() * scale
                ax = axes[row, column]
                ax.plot([pos[v] for v in means.index], means.values, color=PROTOCOL_COLOR[protocol], linewidth=1.1,
                        label=protocol)
                style.ring(ax, [pos[v] for v in means.index], means.values, PROTOCOL_COLOR[protocol], size=12)
        axes[row, 0].set_ylabel(f"{NAMES[arch]}\nAccuracy (%)", fontsize=7)
        axes[row, 1].set_ylabel("Probe R (%)", fontsize=7)
        axes[row, 2].set_ylabel("Adversary train acc. (%)", fontsize=7)
    for ax in axes[-1]:
        ax.set_xticks(range(len(lambdas)), [f"{v:g}" for v in lambdas], fontsize=6.5)
        ax.set_xlabel("Reversal strength $\\lambda$", fontsize=7)
    title(axes[0, 0], "Test accuracy")
    title(axes[0, 1], "Independent probe")
    title(axes[0, 2], "Adversary, final epoch")
    axes[0, 0].legend(fontsize=6.5)
    figure.tight_layout(h_pad=0.8, w_pad=0.8)
    save(figure, "grl_full")


# ---------------------------------------------------------------- Amendment 8
def fig_leace_full() -> None:
    leace = pd.read_csv(F / "tables" / "leace_runs.csv")
    ranks = sorted(leace["rank"].unique())
    pos = {v: i for i, v in enumerate(ranks)}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for arch in ORDER:
        g = leace.loc[leace.architecture == arch]
        means = g.groupby(["protocol", "rank"])[["accuracy", "within_class_accuracy"]].mean() * 100
        r = (means.loc["P1", "within_class_accuracy"] + means.loc["P2", "within_class_accuracy"]) / 2
        gap = means.loc["P1", "accuracy"] - means.loc["P2", "accuracy"]
        p2 = means.loc["P2", "accuracy"] - means.loc[("P2", 0), "accuracy"]
        for ax, values in zip(axes, (r, gap, p2), strict=True):
            ax.plot([pos[v] for v in values.index], values.values, color=style.GRAY_SERIES, linewidth=0.8)
    means = leace.groupby(["protocol", "rank"])[["accuracy", "within_class_accuracy"]].mean() * 100
    pooled = ((means.loc["P1", "within_class_accuracy"] + means.loc["P2", "within_class_accuracy"]) / 2,
              means.loc["P1", "accuracy"] - means.loc["P2", "accuracy"],
              means.loc["P2", "accuracy"] - means.loc[("P2", 0), "accuracy"])
    for ax, values, label in zip(axes, pooled, ("Recoverability R (%)", "P1 minus P2 gap (pp)", "P2 accuracy change (pp)"),
                                 strict=True):
        ax.plot([pos[v] for v in values.index], values.values, color=style.BLUE, linewidth=1.6)
        style.ring(ax, [pos[v] for v in values.index], values.values, style.BLUE, size=20)
        ax.set_xticks(range(len(ranks)), [str(v) for v in ranks])
        ax.set_xlabel("Erasure rank r")
        ax.set_ylabel(label, fontsize=7.5)
    axes[1].set_ylim(-0.2, 2.4)
    style.zero_line(axes[2], "horizontal")
    title(axes[0], "Fingerprint erased")
    title(axes[1], "Gap unchanged")
    title(axes[2], "Accuracy kept")
    figure.tight_layout(w_pad=1.0)
    save(figure, "leace_full")


def sweep_figure(table, level, levels_label, name, archs, title_text):
    figure, axes = plt.subplots(len(archs), 2, figsize=(7.2, 1.9 * len(archs) + 0.4), sharex=True)
    axes = np.atleast_2d(axes)
    levels = sorted(table[level].unique())
    pos = {v: i for i, v in enumerate(levels)}
    for row, arch in enumerate(archs):
        group = table.loc[table.architecture == arch]
        for protocol in ("P1", "P2"):
            sub = group.loc[group.protocol == protocol]
            for column, metric in enumerate(("accuracy", "within_class_accuracy")):
                means = sub.groupby(level)[metric].mean() * 100
                ax = axes[row, column]
                ax.plot([pos[v] for v in means.index], means.values, color=PROTOCOL_COLOR[protocol], linewidth=1.1,
                        label=protocol)
                style.ring(ax, [pos[v] for v in means.index], means.values, PROTOCOL_COLOR[protocol], size=12)
        axes[row, 0].set_ylabel(f"{NAMES[arch]}\nAccuracy (%)", fontsize=7)
        axes[row, 1].set_ylabel("Probe R (%)", fontsize=7)
    for ax in axes[-1]:
        ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
        ax.set_xlabel(levels_label, fontsize=7)
    axes[0, 0].legend(fontsize=6.5)
    title(axes[0, 0], title_text)
    figure.tight_layout(h_pad=0.6, w_pad=0.8)
    save(figure, name)


def fig_ablation_arch() -> None:
    abl = pd.read_csv(F / "tables" / "ablation_runs.csv")
    names = {"original": "orig.", "grey": "grey", "stain_transfer": "stain", "low_pass": "low-pass",
             "illumination_flat": "flat", "background_grey": "no bg"}
    archs = [a for a in ORDER if a in set(abl.architecture)]
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    for ax, arch in zip(axes, archs, strict=True):
        for leak, color, label in ((False, style.BLUE, "clean"), (True, style.ORANGE, "leak")):
            group = abl.loc[(abl.architecture == arch) & (abl.cell.str.startswith("leak") == leak)]
            means = [100 * group.loc[group.ablation == a, "accuracy"].mean() for a in names]
            ax.scatter(range(len(names)), means, s=18, color=color, label=label, zorder=3, edgecolors="white",
                       linewidths=0.7)
        ax.set_xticks(range(len(names)), list(names.values()), rotation=60, fontsize=6.3)
        ax.grid(axis="x", visible=False)
        ax.set_title(NAMES[arch], fontsize=7.5, loc="left", color=style.INK)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(fontsize=6.5, loc="lower right")
    figure.tight_layout(w_pad=0.6)
    save(figure, "ablation_arch")


def fig_carrier_arch() -> None:
    carrier = pd.read_csv(F / "tables" / "carrier_runs.csv")
    carrier["div"] = carrier.cell.str.split("_").str[1]
    wide = carrier.pivot_table(index=["architecture", "seed", "div", "condition"],
                               columns=carrier.cell.str.startswith("leak"), values="accuracy")
    effect = (100 * (wide[True] - wide[False])).rename("effect").reset_index()
    archs = [a for a in ORDER if a in set(carrier.architecture)]
    conditions = (("standard", style.GRAY_SERIES), ("masked", style.BLUE), ("colour", style.ORANGE))
    figure, ax = plt.subplots(figsize=(7.2, 2.7))
    for position, arch in enumerate(archs):
        for shift, (condition, color) in zip((-0.2, 0.0, 0.2), conditions, strict=True):
            mean, low, high = t_interval(effect.loc[(effect.architecture == arch) & (effect.condition == condition), "effect"])
            ax.plot([low, high], [position + shift] * 2, color=color, linewidth=1.1)
            style.ring(ax, [mean], [position + shift], color, size=20, label=condition if position == 0 else None)
    style.zero_line(ax)
    ax.set_yticks(range(len(archs)), [NAMES[a] for a in archs])
    ax.set_xlabel("Leak minus clean accuracy (pp), 20 seed-diversity pairs")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    title(ax, "Carrier factorial on fresh seeds, per architecture")
    figure.tight_layout()
    save(figure, "carrier_arch")


def fig_gradcam() -> None:
    cams = pd.read_csv(F / "tables" / "gradcam_runs.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    archs = [a for a in ORDER if a in set(cams.architecture)]
    for position, arch in enumerate(archs):
        for shift, (leak, color, label) in zip((-0.15, 0.15), ((True, style.ORANGE, "leak"), (False, style.BLUE, "clean")),
                                               strict=True):
            values = 100 * cams.loc[(cams.architecture == arch) & (cams.cell.str.startswith("leak") == leak),
                                    "background_attribution_fraction"]
            jitter = np.random.default_rng(position).uniform(-0.07, 0.07, len(values))
            ax.scatter(values, position + shift + jitter, s=8, color=color, alpha=0.5, linewidths=0)
            style.ring(ax, [values.mean()], [position + shift], color, size=26, label=label if position == 0 else None)
        pixels = 100 * cams.loc[cams.architecture == arch, "background_pixel_fraction"].mean()
        ax.plot([pixels, pixels], [position - 0.35, position + 0.35], color=style.INK_2, linewidth=1)
    ax.set_yticks(range(len(archs)), [NAMES[a] for a in archs])
    ax.set_xlabel("Grad-CAM mass outside the cell (%); bar = background pixel share")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper left")
    title(ax, "Where models look: identical")
    ax = axes[1]
    ax.hist(cams.random_head_spearman, bins=20, color=style.BLUE)
    ax.set_xlabel("Median Spearman: trained vs random head")
    ax.set_ylabel("Runs")
    title(ax, "Sanity check")
    figure.tight_layout(w_pad=1.2)
    save(figure, "gradcam")


def fig_cric_context() -> None:
    cric = json.loads((F / "cric_context.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    items = sorted(cric["per_architecture_secondary"], key=lambda r: r["estimate"])
    for position, item in enumerate(items):
        ax.plot(item["ci95"], [position, position], color=style.INK_2, linewidth=1.1)
        style.ring(ax, [item["estimate"]], [position], style.INK_2, size=22)
    style.zero_line(ax)
    ax.set_yticks(range(len(items)), [NAMES[i["architecture"]] for i in items])
    ax.set_xlabel("Context-class probability, P1 minus P2 (pp)")
    ax.grid(axis="y", visible=False)
    title(ax, "F1 by architecture")
    ax = axes[1]
    entries = (("concordant", cric["accuracy_difference_concordant"]), ("discordant", cric["accuracy_difference_discordant"]))
    for position, (label, record) in enumerate(entries):
        ax.barh(position, record["estimate"], color=style.BLUE, height=0.55)
        ax.plot(record["ci95"], [position, position], color=style.INK, linewidth=1.1)
    ax.set_yticks(range(2), [e[0] for e in entries])
    ax.set_xlabel("Accuracy gain, P1 minus P2 (pp)")
    ax.grid(axis="y", visible=False)
    title(ax, "Gain does not depend on agreement")
    figure.tight_layout(w_pad=1.2)
    save(figure, "cric_context")


def fig_overlap() -> None:
    overlap = json.loads((F / "exploratory" / "overlap.json").read_text())
    figure, ax = plt.subplots(figsize=(7.2, 2.8))
    items = overlap["summary"]
    labels, position = [], 0
    for condition in dict.fromkeys(i["condition"] for i in items):
        for subset, color in (("overlapping", style.ORANGE), ("not overlapping", style.BLUE)):
            item = next(i for i in items if i["condition"] == condition and i["subset"] == subset)
            ax.plot(item["ci95"], [position, position], color=color, linewidth=1.1)
            style.ring(ax, [item["leak_minus_clean_pp"]], [position], color, size=20,
                       label=subset if len(labels) < 2 else None)
            labels.append(f"{condition}, {subset}")
            position += 1
    style.zero_line(ax)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("Leak minus clean accuracy (pp)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=7)
    title(ax, "Exploratory: the leak is not carried by shared pixels")
    figure.tight_layout()
    save(figure, "overlap")


def fig_probe_agreement() -> None:
    agreement = pd.read_csv(F / "tables" / "probe_agreement.csv")
    figure, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(100 * agreement.sklearn, 100 * agreement.device, s=8, color=style.BLUE, alpha=0.6, linewidths=0)
    lo = 100 * min(agreement.sklearn.min(), agreement.device.min()) - 2
    hi = 100 * max(agreement.sklearn.max(), agreement.device.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=style.BASELINE, linewidth=0.8)
    ax.set_xlabel("Registered sklearn probe R (%)")
    ax.set_ylabel("Device L-BFGS probe R (%)")
    title(ax, f"r = {agreement.sklearn.corr(agreement.device):.3f} over {len(agreement)} runs")
    figure.tight_layout()
    save(figure, "probe_agreement")


def fig_structure() -> None:
    from cytofair.risk import structural_risk
    names = (("SIPaKMeD", "provenance/sipakmed_parent_map_p3.parquet"), ("Cx22", "provenance/cx22_parent_map.parquet"),
             ("CRIC", "provenance/cric_parent_map.parquet"), ("Herlev", "provenance/herlev_parent_map.parquet"))
    risks = {name: structural_risk(pd.read_parquet(ROOT / path)) for name, path in names}
    metrics = (("expected_parent_straddle", "Parents straddling (%)", 100),
               ("expected_test_crops_with_training_sibling", "Test obs. with sibling (%)", 100),
               ("expected_training_siblings_per_test_crop", "Training siblings per test obs.", 1),
               ("training_diversity_ratio", "Diversity ratio (crop / parent split)", 1))
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    for ax, (key, label, scale) in zip(axes, metrics, strict=True):
        values = [scale * risks[n][key] for n, _ in names]
        ax.barh(range(len(names)), values, color=style.BLUE, height=0.6)
        for position, value in enumerate(values):
            ax.annotate(f"{value:.1f}" if scale == 100 else f"{value:.2f}", (value, position), xytext=(2, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color=style.INK_2)
        ax.set_xlabel(label, fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), [n for n, _ in names])
    axes[0].invert_yaxis()
    figure.tight_layout(w_pad=0.8)
    save(figure, "structure")


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_timeline()
    fig_gpu_schedule()
    fig_runtime()
    hist = phase1_histories()
    fig_training_curves(hist)
    fig_phase1_points()
    fig_seed_heatmap()
    fig_calibration()
    fig_confusion()
    fig_cric_recall()
    fig_cross_study()
    fig_factorial_cells()
    fig_transplant_arch()
    fig_grl_full()
    fig_leace_full()
    sweep_figure(pd.read_csv(F / "tables" / "linadv_recoverability.csv"), "adversary_lambda", "Adversary strength $\\lambda$",
                 "linadv_full", ["resnet50", "convnext_tiny", "densenet121"],
                 "Linear adversary, full sweep (including degenerate $\\lambda$)")
    sweep_figure(pd.read_csv(F / "tables" / "normalization_recoverability.csv"), "strength", "Normalisation strength s",
                 "normalization_full", ["resnet50", "convnext_tiny", "densenet121", "efficientnet_b0"],
                 "Stain normalisation sweep")
    fig_ablation_arch()
    fig_carrier_arch()
    fig_gradcam()
    fig_cric_context()
    fig_overlap()
    fig_probe_agreement()
    fig_structure()


if __name__ == "__main__":
    main()
