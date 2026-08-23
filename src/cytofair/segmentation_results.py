"""Paper-ready summaries for the preregistered Cx22 segmentation extension."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .statistics import (
    apply_bh_family,
    mean_sd,
    paired_contrasts,
    ranking_stability,
    verify_protocol_order,
)

TASK_METRICS = {
    "semantic_segmentation": (
        "macro_dice",
        "cytoplasm_dice",
        "nucleus_dice",
        "macro_iou",
        "cytoplasm_iou",
        "nucleus_iou",
    ),
    "instance_segmentation": ("instance_dice", "instance_iou", "recall70"),
}


def read_segmentation_runs(paths: list[str | Path]) -> pd.DataFrame:
    records = []
    for source in paths:
        path = Path(source)
        record = json.loads(path.read_text())
        record["result_path"] = str(path.resolve())
        records.append(record)
    if not records:
        raise ValueError("No segmentation results supplied")
    frame = pd.DataFrame(records)
    keys = ["task", "architecture", "protocol", "seed"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Segmentation runs contain missing or duplicate experiment keys")
    unknown = sorted(set(frame.task) - set(TASK_METRICS))
    if unknown:
        raise ValueError(f"Unknown segmentation tasks: {unknown}")
    return frame


def summarize_segmentation(
    paths: list[str | Path], *, bootstrap_repetitions: int = 10_000
) -> dict[str, pd.DataFrame]:
    runs = read_segmentation_runs(paths)
    summaries, primary_frames, secondary_frames, ordering_frames, ranking_frames = [], [], [], [], []
    for task, metrics in TASK_METRICS.items():
        selected = runs.loc[runs.task == task].copy()
        if selected.empty:
            continue
        missing = sorted(set(metrics) - set(selected.columns))
        if missing:
            raise ValueError(f"{task} results miss metrics: {missing}")
        for metric in metrics:
            summary = mean_sd(
                selected,
                value=metric,
                by=["dataset", "task", "architecture", "protocol"],
            )
            summary.insert(4, "metric", metric)
            summary["mean_sd"] = summary.apply(
                lambda row: f"{row['mean']:.4f} ± {row['sd']:.4f}"
                if row.complete
                else "incomplete",
                axis=1,
            )
            summaries.append(summary)
        primary_metric = metrics[0]
        confirmatory = selected.loc[selected.protocol != "OFFICIAL"]
        family = []
        for left in ("P1", "P1-prime"):
            family.append(
                paired_contrasts(
                    confirmatory,
                    value=primary_metric,
                    family_columns=["task", "architecture"],
                    left_protocol=left,
                    right_protocol="P2",
                    bootstrap_repetitions=bootstrap_repetitions,
                    apply_bh=False,
                )
            )
        primary = apply_bh_family(pd.concat(family, ignore_index=True))
        primary["multiplicity_family"] = f"cx22:{task}:P1/P1-prime-v-P2"
        primary_frames.append(primary)
        secondary = paired_contrasts(
            confirmatory,
            value=primary_metric,
            family_columns=["task", "architecture"],
            left_protocol="P2",
            right_protocol="P3",
            bootstrap_repetitions=bootstrap_repetitions,
            apply_bh=False,
        ).drop(columns="q_value")
        secondary_frames.append(secondary)
        ordering_frames.append(
            verify_protocol_order(
                confirmatory, value=primary_metric, by=["task", "architecture"]
            )
        )
        ranking_frames.append(
            ranking_stability(
                confirmatory,
                value=primary_metric,
                dataset_column="task",
                bootstrap_repetitions=bootstrap_repetitions,
            )
        )
    return {
        "summary": pd.concat(summaries, ignore_index=True),
        "primary": pd.concat(primary_frames, ignore_index=True),
        "secondary": pd.concat(secondary_frames, ignore_index=True),
        "ordering": pd.concat(ordering_frames, ignore_index=True),
        "ranking": pd.concat(ranking_frames, ignore_index=True),
    }
