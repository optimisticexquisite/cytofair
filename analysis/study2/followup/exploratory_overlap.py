#!/usr/bin/env python3
"""EXPLORATORY (not registered): is the 2C leak advantage carried by literally shared pixels?

In leak cells a third or more of the common test crops share source pixels with a
training sibling (overlapping bounding boxes in the same field). For each diversity
level, test crops are split by whether they overlap a training crop of the leak cell,
and the leak-minus-clean accuracy difference is computed on each subset, for:

* the registered Study 2 2C models on original images and after test-time stain
  transfer (Amendment 8, F2 logits), and
* the Amendment 8 carrier factorial (standard, background-masked, colour-augmented).

The unit of the interval is the (architecture, seed) pair: per pair and diversity
level, leak minus clean accuracy on each subset; means with 95% t intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def overlap_flags(split: pd.DataFrame, boxes: pd.DataFrame) -> pd.Series:
    """Per test crop: does it share any source pixel with a training crop of the same parent?"""
    frame = split.merge(boxes, on="crop_id", how="left")
    flags = {}
    for _, group in frame.groupby("parent_id"):
        train = group.loc[group.partition == "train"]
        for row in group.loc[group.partition == "test"].itertuples(index=False):
            hit = False
            if len(train):
                hit = bool((
                    (np.maximum(train.x, row.x) < np.minimum(train.x + train.width, row.x + row.width))
                    & (np.maximum(train.y, row.y) < np.minimum(train.y + train.height, row.y + row.height))
                ).any())
            flags[row.crop_id] = hit
    return pd.Series(flags, name="overlap")


def correctness_from_predictions(run: dict) -> pd.Series:
    predictions = pd.read_parquet(run["artifacts"]["predictions"]["path"])
    return (predictions.prediction == predictions.label).set_axis(predictions.crop_id)


def correctness_from_ablation(record: dict, ablation: str) -> pd.Series:
    source = json.loads(Path(record["source_run"]).read_text())
    classes = pd.read_csv(source["split_path"]).set_index("crop_id").class_name
    index = {name: i for i, name in enumerate(source["class_names"])}
    archive = np.load(record["artifacts"]["ablation_logits"]["path"])
    ids = pd.Series(archive["crop_id"].astype(str))
    keep = ids.str.endswith(f"|{ablation}").to_numpy()
    crops = ids[keep].str.rsplit("|", n=1).str[0].to_numpy()
    predicted = archive["logits"][keep].argmax(axis=1)
    return pd.Series(predicted == classes.loc[crops].map(index).to_numpy(), index=crops)


def summarise(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    out = []
    for (condition, subset), group in frame.groupby(["condition", "subset"]):
        values = group.difference.to_numpy()
        half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        out.append({"condition": condition, "subset": subset, "pairs": len(values),
                    "mean_crops": float(group.crops.mean()), "leak_minus_clean_pp": float(values.mean()),
                    "ci95": [float(values.mean() - half), float(values.mean() + half)]})
    return out


def contrasts(splits_root: Path, records: dict, boxes: pd.DataFrame, condition: str, getter) -> list[dict]:
    rows = []
    for (architecture, seed, div), cells in records.items():
        if set(cells) != {"leak", "clean"}:
            continue
        split = pd.read_csv(splits_root / f"sipakmed_2c_leak_{div}_seed{seed}.csv")
        flags = overlap_flags(split.loc[split.partition.isin(["train", "test"])], boxes)
        leak, clean = getter(cells["leak"]), getter(cells["clean"])
        for subset, crops in (("overlapping", flags.index[flags]), ("not overlapping", flags.index[~flags])):
            rows.append({"condition": condition, "subset": subset, "architecture": architecture, "seed": seed,
                         "div": div, "crops": len(crops),
                         "difference": 100 * (leak.loc[crops].mean() - clean.loc[crops].mean())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study2-runs", required=True, help="Study 2 run root (runs/factorial/runs)")
    parser.add_argument("--followup-runs", required=True, help="Amendment 8 run root (runs/followup)")
    parser.add_argument("--boxes", default="provenance/sipakmed_branch_b.parquet")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    boxes = pd.read_parquet(args.boxes)[["crop_id", "x", "y", "width", "height"]]
    rows = []

    study2, ablations = {}, {}
    for path in sorted(Path(args.study2_runs, "factorial", "runs").glob("*.json")):
        run = json.loads(path.read_text())
        leak, div = run["protocol"].split("_")
        study2.setdefault((run["architecture"], run["seed"], div), {})[leak] = run
    for path in sorted(Path(args.followup_runs, "factorial_eval", "runs").glob("*.json")):
        record = json.loads(path.read_text())
        leak, div = record["protocol"].split("_")
        ablations.setdefault((record["architecture"], record["seed"], div), {})[leak] = record
    root2c = Path("splits/study2/sipakmed_2c")
    rows += contrasts(root2c, study2, boxes, "Study 2, original", correctness_from_predictions)
    rows += contrasts(root2c, ablations, boxes, "Study 2, stain transfer",
                      lambda record: correctness_from_ablation(record, "stain_transfer"))

    carrier = {}
    for path in sorted(Path(args.followup_runs, "carrier", "runs").glob("*.json")):
        run = json.loads(path.read_text())
        leak, div = run["protocol"].split("_")
        condition = path.stem.rsplit("_", 1)[1]  # standard / masked / colour
        carrier.setdefault(condition, {}).setdefault((run["architecture"], run["seed"], div), {})[leak] = run
    for condition, records in sorted(carrier.items()):
        rows += contrasts(Path("splits/study2/followup/sipakmed_2c"), records, boxes, f"carrier, {condition}",
                          correctness_from_predictions)

    out = {"note": "EXPLORATORY, not registered", "summary": summarise(rows)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(Path(args.output).with_suffix(".csv"), index=False)
    for item in out["summary"]:
        print(f"{item['condition']:<26} {item['subset']:<16} {item['leak_minus_clean_pp']:6.2f} "
              f"[{item['ci95'][0]:.2f}, {item['ci95'][1]:.2f}]  pairs={item['pairs']} crops~{item['mean_crops']:.0f}")


if __name__ == "__main__":
    main()
