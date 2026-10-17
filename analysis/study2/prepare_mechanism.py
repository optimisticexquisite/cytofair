#!/usr/bin/env python3
"""Reduce Study 2 artifacts to the per-run and per-crop tables the registered models use.

* 2B: per factorial run, accuracy on every transplant condition and the donor-
  class probability shift for other-class backgrounds (all recipients and the
  registered background-fraction >= 0.30 subset).
* 2A: per gradient-reversal run, within-class parent recoverability of the
  frozen penultimate features on the balanced probe split.
* Phase 1: per crop, accuracy when the crop is tested under P1 versus P2, its
  number of training siblings, and literal source-pixel overlap in each P1 split.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cytofair.recoverability import parent_recoverability


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def transplant_metrics(run: dict, plan: pd.DataFrame, class_names: list[str]) -> list[dict]:
    artifact = run["artifacts"]["extra:transplants"]
    archive = np.load(artifact["path"])
    logits = pd.DataFrame(archive["logits"], index=archive["crop_id"].astype(str))
    plan = plan.set_index("composite_id")
    probabilities = pd.DataFrame(_softmax(logits.loc[plan.index].to_numpy()), index=plan.index, columns=class_names)
    plan = plan.assign(
        prediction=probabilities.to_numpy().argmax(axis=1),
        true_index=plan.recipient_class.map({name: i for i, name in enumerate(class_names)}),
        donor_index=plan.donor_class.map({name: i for i, name in enumerate(class_names)}),
    )
    plan["correct"] = plan.prediction == plan.true_index
    plan["p_donor"] = probabilities.to_numpy()[np.arange(len(plan)), plan.donor_index.to_numpy()]
    plan["flip_to_donor"] = plan.prediction == plan.donor_index
    rows = []
    for subset, frame in (("all", plan), ("background_ge_0.30", plan.loc[plan.background_fraction >= 0.30])):
        accuracy = frame.groupby("condition").correct.mean()
        # Probability of the donor class under an other-class background minus the
        # probability of that same class under the recipient's unseen same-class
        # background: a pure background swap with matched seams and inpainting.
        same = frame.loc[frame.condition == "unseen_same_class"]
        baseline = probabilities.loc[same.index].set_axis(same.recipient_id.to_numpy())
        other = frame.loc[frame.condition == "unseen_other_class"].copy()
        other["p_donor_baseline"] = [
            baseline.loc[recipient, donor]
            for recipient, donor in zip(other.recipient_id, other.donor_class, strict=True)
        ]
        rows.append(
            {
                "subset": subset,
                "recipients": frame.recipient_id.nunique(),
                **{f"acc_{condition}": float(value) for condition, value in accuracy.items()},
                "donor_probability_shift": float((other.p_donor - other.p_donor_baseline).mean()),
                "flip_to_donor_rate": float(other.flip_to_donor.mean()),
            }
        )
    return rows


def recoverability_row(path: str, probe_path: str, logistic: bool = True) -> dict:
    run = json.loads(Path(path).read_text())
    probe = pd.read_csv(probe_path)
    archive = np.load(run["artifacts"]["probe_features"]["path"])
    metrics = parent_recoverability(
        archive["features"].astype(np.float32), archive["crop_id"].astype(str), probe, logistic=logistic
    )
    seen = set(pd.read_csv(run["split_path"]).query("partition == 'train'").parent_id)
    probe_parents = set(probe.parent_id)
    return {
        "run_json": path,
        **metrics,
        "probe_parents_seen_in_training": len(probe_parents & seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="collected runs directory (from collect_runs.py)")
    parser.add_argument("--splits", default="splits/study2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    runs_dir, splits, output = Path(args.runs), Path(args.splits), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    factorial_path = runs_dir / "factorial_runs.csv"
    if factorial_path.is_file():
        factorial = pd.read_csv(factorial_path)
        rows = []
        for item in factorial.itertuples(index=False):
            run = json.loads(Path(item.run_json).read_text())
            plan = pd.read_csv(splits / "sipakmed_2b" / f"sipakmed_2b_plan_seed{item.seed}.csv")
            backgrounds = pd.read_csv(splits / "sipakmed_2b" / f"background_fraction_seed{item.seed}.csv")
            plan = plan.merge(backgrounds, on="recipient_id", how="left", validate="many_to_one")
            for metrics in transplant_metrics(run, plan, run["class_names"]):
                rows.append({"architecture": item.architecture, "seed": item.seed, "cell": item.protocol,
                             "accuracy": item.accuracy, **metrics})
        pd.DataFrame(rows).to_csv(output / "transplant_runs.csv", index=False)

    grl_path = runs_dir / "grl_runs.csv"
    if grl_path.is_file():
        grl = pd.read_csv(grl_path)
        probe = str(splits / "sipakmed_probe_split.csv")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            metrics = list(pool.map(recoverability_row, grl.run_json, [probe] * len(grl)))
        pd.DataFrame(metrics).merge(grl, on="run_json").to_csv(output / "grl_recoverability.csv", index=False)

    phase1_path = runs_dir / "phase1_runs.csv"
    if phase1_path.is_file():
        phase1 = pd.read_csv(phase1_path)
        probe = str(splits / "sipakmed_probe_split.csv")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            metrics = list(pool.map(recoverability_row, phase1.run_json, [probe] * len(phase1), [False] * len(phase1)))
        pd.DataFrame(metrics).merge(phase1, on="run_json").to_csv(output / "phase1_recoverability.csv", index=False)
        boxes = pd.read_parquet("provenance/sipakmed_branch_b.parquet")[["crop_id", "x", "y", "width", "height"]]
        records = []
        for item in phase1.loc[phase1.protocol.isin(["P1", "P2"])].itertuples(index=False):
            run = json.loads(Path(item.run_json).read_text())
            predictions = pd.read_parquet(run["artifacts"]["predictions"]["path"])
            predictions["correct"] = predictions.label == predictions.prediction
            predictions = predictions[["crop_id", "parent_id", "label", "correct"]]
            predictions["architecture"], predictions["seed"], predictions["protocol"] = (
                item.architecture, item.seed, item.protocol,
            )
            records.append(predictions)
        crops = pd.concat(records, ignore_index=True)
        exposure = []
        for seed in sorted(crops.seed.unique()):
            split = pd.read_csv(splits / "sipakmed" / f"sipakmed_p1_seed{seed}.csv")
            train = split.loc[split.partition == "train"].merge(boxes, on="crop_id")
            test = split.loc[split.partition == "test"].merge(boxes, on="crop_id")
            siblings = train.groupby("parent_id").size()
            by_parent = {parent: group for parent, group in train.groupby("parent_id")}
            for row in test.itertuples(index=False):
                group = by_parent.get(row.parent_id)
                overlap = False
                if group is not None:
                    overlap = bool(
                        (
                            (np.maximum(group.x, row.x) < np.minimum(group.x + group.width, row.x + row.width))
                            & (np.maximum(group.y, row.y) < np.minimum(group.y + group.height, row.y + row.height))
                        ).any()
                    )
                exposure.append({"seed": seed, "crop_id": row.crop_id, "train_siblings": int(siblings.get(row.parent_id, 0)),
                                 "literal_overlap": overlap})
        crops = crops.merge(pd.DataFrame(exposure).assign(protocol="P1"), on=["seed", "crop_id", "protocol"], how="left")
        crops.to_parquet(output / "phase1_crop_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
