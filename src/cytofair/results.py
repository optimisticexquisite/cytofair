"""Paper-ready summaries for preregistered model runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .statistics import apply_bh_family, mean_sd, paired_contrasts, verify_protocol_order


def read_json_runs(paths: list[str | Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        record["result_path"] = str(Path(path).resolve())
        records.append(record)
    if not records:
        raise ValueError("No run results supplied")
    frame = pd.DataFrame(records)
    keys = ["dataset", "architecture", "protocol", "seed"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Run results contain missing or duplicate experiment keys")
    return frame


def summarize_classification(
    paths: list[str | Path], *, bootstrap_repetitions: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mean/SD, the single confirmatory BH family, secondary P2/P3, and order."""
    runs = read_json_runs(paths)
    required = {"accuracy", "macro_f1", "balanced_accuracy"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Classification results miss metrics: {missing}")
    summaries = []
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        summary = mean_sd(
            runs,
            value=metric,
            by=["dataset", "architecture", "protocol"],
        )
        summary.insert(3, "metric", metric)
        summary["mean_sd"] = summary.apply(
            lambda row: f"{row['mean']:.4f} ± {row['sd']:.4f}" if row.complete else "incomplete",
            axis=1,
        )
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)

    primary_parts = []
    for left in ("P1", "P1-prime"):
        part = paired_contrasts(
            runs,
            value="accuracy",
            family_columns=["dataset", "architecture"],
            left_protocol=left,
            right_protocol="P2",
            bootstrap_repetitions=bootstrap_repetitions,
            apply_bh=False,
        )
        primary_parts.append(part)
    primary = apply_bh_family(pd.concat(primary_parts, ignore_index=True))
    primary["multiplicity_family"] = "classification:P1-v-P2_and_P1-prime-v-P2"

    secondary = paired_contrasts(
        runs,
        value="accuracy",
        family_columns=["dataset", "architecture"],
        left_protocol="P2",
        right_protocol="P3",
        bootstrap_repetitions=bootstrap_repetitions,
        apply_bh=False,
    ).drop(columns="q_value")
    ordering = verify_protocol_order(
        runs, value="accuracy", by=["dataset", "architecture"]
    )
    return summary, primary, secondary, ordering
