#!/usr/bin/env python3
"""F1 (Amendment 8): does crop-level training pull CRIC predictions toward the slide's majority class?

CRIC slide images contain cells of several classes, so a test cell's training
siblings can carry a label different from its own. For every cell, the
*context class* is the most frequent class among the other cells of its slide
(cells whose slide has no other cell or a tied majority are excluded). A cell is
*discordant* if its own class differs from the context class.

If crop-level (P1) models use slide context, discordant cells should receive more
probability on the context class when tested under P1 (their slide is in
training) than under P2 (their slide is unseen). The within-cell difference
controls cell difficulty; inference is a cluster bootstrap over slides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPETITIONS = 10_000
BOOTSTRAP_SEED = 20261019


def context_classes(provenance: pd.DataFrame) -> pd.DataFrame:
    """Context class of every cell from the other cells of its slide (leave-one-out)."""
    counts = provenance.groupby(["parent_id", "class_name"]).size().unstack(fill_value=0)
    rows = []
    for cell in provenance.itertuples(index=False):
        others = counts.loc[cell.parent_id].copy()
        others[cell.class_name] -= 1
        if others.sum() == 0:
            continue
        top = others.max()
        winners = others.index[others == top]
        if len(winners) != 1:
            continue
        rows.append({"crop_id": cell.crop_id, "parent_id": cell.parent_id, "class_name": cell.class_name,
                     "context_class": winners[0], "slide_cells": int(counts.loc[cell.parent_id].sum())})
    frame = pd.DataFrame(rows)
    frame["discordant"] = frame.class_name != frame.context_class
    return frame


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def cell_table(runs: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Per-run, per-test-cell probability of the context class and correctness."""
    records = []
    for item in runs.itertuples(index=False):
        run = json.loads(Path(item.run_json).read_text())
        predictions = pd.read_parquet(run["artifacts"]["predictions"]["path"])
        logit_columns = [f"logit_{index}" for index in range(len(run["class_names"]))]
        probabilities = softmax(predictions[logit_columns].to_numpy(dtype=np.float64))
        frame = predictions[["crop_id", "label", "prediction"]].merge(context, on="crop_id", how="inner")
        index = {name: position for position, name in enumerate(run["class_names"])}
        rows = predictions.reset_index(drop=True).reset_index().set_index("crop_id").loc[frame.crop_id, "index"]
        frame["p_context"] = probabilities[rows.to_numpy(), frame.context_class.map(index).to_numpy()]
        frame["predicted_context"] = frame.prediction == frame.context_class.map(index)
        frame["correct"] = frame.prediction == frame.label
        frame["architecture"], frame["seed"], frame["protocol"] = item.architecture, item.seed, item.protocol
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def cluster_bootstrap(values: pd.DataFrame, column: str, *, groups: str = "parent_id") -> dict:
    """Mean of ``column`` with a percentile cluster bootstrap over slides."""
    estimate = float(values[column].mean())
    sums = values.groupby(groups)[column].agg(["sum", "count"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(sums), size=(REPETITIONS, len(sums)))
    totals = sums["sum"].to_numpy()[draws].sum(axis=1)
    counts = sums["count"].to_numpy()[draws].sum(axis=1)
    boot = totals / counts
    return {
        "estimate": estimate,
        "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "p_two_sided": float(min(1.0, 2.0 * min((boot <= 0).mean(), (boot >= 0).mean()))),
        "cells": len(values),
        "slides": len(sums),
        "replicates": REPETITIONS,
    }


def contrast_bootstrap(values: pd.DataFrame, column: str, flag: str, *, groups: str = "parent_id") -> dict:
    """Mean of ``column`` where ``flag`` is false minus where it is true, cluster-bootstrapped over slides."""
    estimate = float(values.loc[~values[flag], column].mean() - values.loc[values[flag], column].mean())
    table = values.assign(_off=~values[flag]).groupby([groups, "_off"])[column].agg(["sum", "count"]).unstack(
        "_off", fill_value=0
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(table), size=(REPETITIONS, len(table)))

    def mean(off: bool) -> np.ndarray:
        return table[("sum", off)].to_numpy()[draws].sum(axis=1) / table[("count", off)].to_numpy()[draws].sum(axis=1)

    boot = mean(True) - mean(False)
    return {
        "estimate": estimate,
        "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "p_two_sided": float(min(1.0, 2.0 * min((boot <= 0).mean(), (boot >= 0).mean()))),
        "slides": len(table),
        "replicates": REPETITIONS,
    }


def within_cell(table: pd.DataFrame, column: str) -> pd.DataFrame:
    means = table.groupby(["crop_id", "protocol"])[column].mean().unstack("protocol").dropna(subset=["P1", "P2"])
    info = table.drop_duplicates("crop_id").set_index("crop_id")[["parent_id", "discordant", "class_name"]]
    out = info.loc[means.index].copy()
    out["difference"] = 100.0 * (means["P1"] - means["P2"])
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="cric_runs.csv from the Study 2 collector")
    parser.add_argument("--provenance", default="provenance/cric_parent_map.parquet")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = pd.read_csv(args.runs)
    runs = runs.loc[runs.protocol.isin(["P1", "P2"])]
    if len(runs) != 6 * 10 * 2:
        raise SystemExit(f"expected 120 CRIC P1/P2 runs, found {len(runs)}")
    context = context_classes(pd.read_parquet(args.provenance))
    table = cell_table(runs, context)
    pull = within_cell(table, "p_context")
    accuracy = within_cell(table, "correct")
    discordant = pull.loc[pull.discordant]
    out = {
        "context_cells": len(context),
        "discordant_cells": int(context.discordant.sum()),
        "F1": {
            **cluster_bootstrap(discordant, "difference"),
            "quantity": "within-cell P1 - P2 probability of the slide's context class, discordant cells (pp)",
        },
    }
    # Secondary: the accuracy difference by concordance, and its contrast.
    concordant_accuracy = accuracy.loc[~accuracy.discordant]
    discordant_accuracy = accuracy.loc[accuracy.discordant]
    out["accuracy_difference_concordant"] = cluster_bootstrap(concordant_accuracy, "difference")
    out["accuracy_difference_discordant"] = cluster_bootstrap(discordant_accuracy, "difference")
    out["concordance_contrast_secondary"] = {
        **contrast_bootstrap(accuracy, "difference", "discordant"),
        "quantity": "accuracy difference (P1 - P2) in concordant minus discordant cells (pp)",
    }
    predicted = within_cell(table, "predicted_context")
    out["context_prediction_rate_discordant_secondary"] = cluster_bootstrap(
        predicted.loc[predicted.discordant], "difference"
    )
    per_architecture = []
    for architecture, group in table.groupby("architecture"):
        cells = within_cell(group, "p_context")
        cells = cells.loc[cells.discordant]
        per_architecture.append({"architecture": architecture, **cluster_bootstrap(cells, "difference")})
    out["per_architecture_secondary"] = per_architecture
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["F1"], indent=2))


if __name__ == "__main__":
    main()
