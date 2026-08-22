#!/usr/bin/env python3
"""Render structural and matched-classification CSVs into submission-ready Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROTOCOL_LABEL = {"P1": "P1", "P1-prime": "P1′", "P2": "P2", "P3": "P3"}


def _markdown(headers: list[str], rows: list[list[object]]) -> str:
    escaped = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in rows
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in escaped),
        ]
    )


def _p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def structural_table(frame: pd.DataFrame) -> str:
    required = {
        "dataset",
        "crops",
        "parents",
        "analytic_p1_straddle_pct",
        "observed_p1_straddle_pct_mean_sd",
        "observed_p1_test_leak_pct_mean_sd",
        "observed_p2_straddle_pct_mean_sd",
        "observed_p3_straddle_pct_mean_sd",
        "n_seeds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Structural table misses columns: {missing}")
    if (frame.n_seeds < 5).any():
        raise ValueError("Structural table requires at least five seeds")
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            [
                row.dataset,
                f"{int(row.crops):,}",
                f"{int(row.parents):,}",
                f"{row.analytic_p1_straddle_pct:.2f}%",
                f"{row.observed_p1_straddle_pct_mean_sd}%",
                f"{row.observed_p1_test_leak_pct_mean_sd}%",
                f"{row.observed_p2_straddle_pct_mean_sd}%",
                f"{row.observed_p3_straddle_pct_mean_sd}%",
            ]
        )
    return _markdown(
        [
            "Dataset",
            "Crops",
            "Parents",
            "Analytic P1 straddle",
            "Observed P1 straddle",
            "P1 test crops with train parent",
            "P2 straddle",
            "P3 straddle",
        ],
        rows,
    )


def classification_tables(
    summary: pd.DataFrame, contrasts: pd.DataFrame
) -> tuple[str, str]:
    required_summary = {
        "dataset",
        "architecture",
        "protocol",
        "metric",
        "n",
        "mean_sd",
        "complete",
    }
    missing = sorted(required_summary - set(summary.columns))
    if missing:
        raise ValueError(f"Classification summary misses columns: {missing}")
    accuracy = summary.loc[summary.metric == "accuracy"].copy()
    if accuracy.empty or not accuracy.complete.astype(bool).all() or (accuracy.n < 5).any():
        raise ValueError("Classification table requires complete five-seed accuracy cells")
    wide = accuracy.pivot(
        index=["dataset", "architecture"], columns="protocol", values="mean_sd"
    ).reset_index()
    for protocol in PROTOCOL_LABEL:
        if protocol not in wide:
            raise ValueError(f"Classification table misses protocol {protocol}")
    score_rows = [
        [
            row.dataset,
            row.architecture,
            *(getattr(row, protocol.replace("-", "_")) for protocol in PROTOCOL_LABEL),
        ]
        for row in wide.rename(columns={"P1-prime": "P1_prime"}).itertuples(index=False)
    ]
    scores = _markdown(
        ["Dataset", "Architecture", *PROTOCOL_LABEL.values()],
        score_rows,
    )

    required_contrast = {
        "dataset",
        "architecture",
        "left_protocol",
        "right_protocol",
        "n",
        "mean_difference",
        "sd_difference",
        "test",
        "p_value",
        "q_value",
        "effect_name",
        "effect_size",
        "effect_ci_low",
        "effect_ci_high",
        "multiplicity_family",
    }
    missing = sorted(required_contrast - set(contrasts.columns))
    if missing:
        raise ValueError(f"Contrast table misses columns: {missing}")
    if contrasts.multiplicity_family.nunique() != 1:
        raise ValueError("Contrast table must contain exactly one declared BH family")
    contrast_rows = []
    for row in contrasts.itertuples(index=False):
        if int(row.n) < 5:
            raise ValueError("Contrast table requires at least five paired seeds")
        if pd.isna(row.p_value):
            inference = f"{row.test}; no p-value"
        else:
            if pd.isna(row.q_value) or not row.effect_name or pd.isna(row.effect_size):
                raise ValueError("Every p-value requires BH q and an effect size")
            effect = f"{row.effect_name}={row.effect_size:.3f}"
            if not pd.isna(row.effect_ci_low) and not pd.isna(row.effect_ci_high):
                effect += f" [{row.effect_ci_low:.3f}, {row.effect_ci_high:.3f}]"
            inference = f"p={_p(row.p_value)}, q={_p(row.q_value)}; {effect}"
        contrast_rows.append(
            [
                row.dataset,
                row.architecture,
                f"{PROTOCOL_LABEL[row.left_protocol]}−{PROTOCOL_LABEL[row.right_protocol]}",
                f"{row.mean_difference:.4f} ± {row.sd_difference:.4f}",
                row.test,
                inference,
            ]
        )
    inference = _markdown(
        ["Dataset", "Architecture", "Contrast", "Paired Δ, mean ± SD", "Test", "Inference"],
        contrast_rows,
    )
    return scores, inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural", default="results/table1_structural_leakage.csv")
    parser.add_argument("--classification", default="results/classification_summary.csv")
    parser.add_argument("--contrasts", default="results/classification_primary_contrasts.csv")
    parser.add_argument("--output", default="paper/generated_tables.md")
    args = parser.parse_args()
    structural = structural_table(pd.read_csv(args.structural))
    scores, inference = classification_tables(
        pd.read_csv(args.classification), pd.read_csv(args.contrasts)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Generated paper tables\n\n"
        "All stochastic cells are mean ± sample SD over at least five fixed seeds.\n\n"
        "## Table 1. Structural leakage\n\n"
        f"{structural}\n\n"
        "## Table 2a. Matched classification accuracy\n\n"
        f"{scores}\n\n"
        "## Table 2b. Preregistered paired inference\n\n"
        f"{inference}\n"
    )
    print(output)


if __name__ == "__main__":
    main()
