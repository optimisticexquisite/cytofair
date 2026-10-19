#!/usr/bin/env python3
"""Collect and reduce the Amendment 8 follow-up outputs into the tables ``followup_lmm.R`` fits.

Every registered task must have exactly one valid output whose recorded fields
match the task's ``expect`` block; anything missing or mismatched is listed and
the command fails (``--allow-incomplete`` is for progress reports only). As in
Amendment 7, a linear-adversary run whose every attempt ended in a non-finite loss
is recorded as diverged rather than missing.

Parent recoverability for the linear-adversary, normalisation, and LEACE arms is
the same within-class logistic probe (``cytofair.followup_eval.fit_logistic``,
sklearn's objective with C = 1 on standardised features), so the pooled
dose-response compares like with like.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cytofair.followup_eval import within_class_recoverability

NORMALIZATION_ARCHITECTURES = ("resnet50", "convnext_tiny", "densenet121", "efficientnet_b0")
FOLLOWUP_SPLIT_SEEDS = (601, 653, 701, 751, 809, 853, 907, 953, 1009, 1051)


def divergence_epoch(task: dict, log_dir: str | None) -> int | None:
    if log_dir is None:
        return None
    logs = sorted(Path(log_dir).glob(f"{task['id']}.attempt*.log"))
    epochs = []
    for log in logs:
        match = re.search(r"Non-finite training loss at epoch (\d+)", log.read_text(errors="replace"))
        if match is None:
            return None
        epochs.append(int(match.group(1)))
    return min(epochs) if epochs else None


def collect(task_dir: Path, log_dir: str | None):
    records, problems, diverged = {}, [], []
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            run_path = Path(task["run_json"])
            if not run_path.is_file():
                epoch = divergence_epoch(task, log_dir)
                if task["experiment"] == "linadv" and epoch is not None:
                    diverged.append({"task_id": task["id"], **task["expect"], "diverged_at_epoch": epoch})
                else:
                    problems.append(f"missing: {task['id']}")
                continue
            record = json.loads(run_path.read_text())
            for key, value in task["expect"].items():
                if record.get(key) != value:
                    problems.append(f"{task['id']}: {key}={record.get(key)!r} expected {value!r}")
            records.setdefault(task["experiment"], []).append(record)
    return records, problems, diverged


class Prober:
    def __init__(self, probe_path: str):
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.probe = pd.read_csv(probe_path)

    def __call__(self, features_path: str) -> dict:
        archive = np.load(features_path)
        crop_ids = archive["crop_id"].astype(str)
        features = self.torch.as_tensor(archive["features"].astype(np.float32), device=self.device)
        return within_class_recoverability(features, self.probe, {crop: i for i, crop in enumerate(crop_ids)})


def ablation_tables(records: list[dict]):
    rows, cams = [], []
    for record in records:
        source = json.loads(Path(record["source_run"]).read_text())
        split = pd.read_csv(source["split_path"])
        classes = split.set_index("crop_id").class_name
        index = {name: position for position, name in enumerate(source["class_names"])}
        archive = np.load(record["artifacts"]["ablation_logits"]["path"])
        ids = pd.Series(archive["crop_id"].astype(str))
        parts = ids.str.rsplit("|", n=1, expand=True)
        frame = pd.DataFrame({"recipient_id": parts[0], "ablation": parts[1],
                              "prediction": archive["logits"].argmax(axis=1)})
        frame["correct"] = frame.prediction == frame.recipient_id.map(classes).map(index)
        recorded = pd.read_parquet(source["artifacts"]["predictions"]["path"]).set_index("crop_id").prediction
        original = frame.loc[frame.ablation == "original"]
        agreement = float((original.prediction.to_numpy() == recorded.loc[original.recipient_id].to_numpy()).mean())
        base = {"architecture": record["architecture"], "seed": record["seed"], "cell": record["protocol"]}
        for ablation, group in frame.groupby("ablation"):
            rows.append({**base, "ablation": ablation, "accuracy": float(group.correct.mean()),
                         "crops": len(group), "original_prediction_agreement": agreement})
        if record.get("gradcam"):
            cam = record["gradcam"]
            cams.append({**base, "background_attribution_fraction": cam["mean_background_attribution_fraction"],
                         "background_pixel_fraction": cam["mean_background_pixel_fraction"],
                         "random_head_spearman": cam["median_random_head_spearman"], "crops": cam["crops"]})
    return pd.DataFrame(rows), pd.DataFrame(cams)


def leace_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        for item in record["ranks"]:
            rows.append({"architecture": record["architecture"], "seed": record["seed"],
                         "protocol": record["protocol"], "original_head_accuracy": record["original_head_accuracy"],
                         "recorded_accuracy": record["recorded_accuracy"], **item})
    return pd.DataFrame(rows)


def training_table(records: list[dict], prober: Prober | None = None, **extra) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {key: record.get(key) for key in (
            "architecture", "seed", "protocol", "accuracy", "macro_f1", "balanced_accuracy", "cache_name",
            "color_augment", "adversary_lambda", "adversary_kind", "train_parents", "git_sha",
        )}
        row.update(extra)
        if prober is not None:
            row.update(prober(record["artifacts"]["probe_features"]["path"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="splits/study2/followup/tasks")
    parser.add_argument("--study2-analysis", default="results/study2/analysis")
    parser.add_argument("--probe", default="splits/study2/sipakmed_probe_split.csv")
    parser.add_argument("--logs", help="attempt-log directory (linear-adversary divergence)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, problems, diverged = collect(Path(args.tasks), args.logs)
    report = {"collected": {key: len(value) for key, value in records.items()}, "diverged": len(diverged),
              "problems": problems}
    (output / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}), f"{len(problems)} problems")
    for problem in problems[:20]:
        print(" ", problem)
    if problems and not args.allow_incomplete:
        raise SystemExit(1)
    pd.DataFrame(diverged, columns=["task_id", "dataset", "protocol", "architecture", "seed", "split_file_sha256",
                                    "cache_name", "adversary_lambda", "adversary_kind", "diverged_at_epoch"]
                 ).to_csv(output / "linadv_diverged.csv", index=False)

    prober = Prober(args.probe)
    if "ffact" in records:
        ablations, cams = ablation_tables(records["ffact"])
        ablations.to_csv(output / "ablation_runs.csv", index=False)
        cams.to_csv(output / "gradcam_runs.csv", index=False)
    if "leace" in records:
        leace_table(records["leace"]).to_csv(output / "leace_runs.csv", index=False)
    if "carrier" in records:
        carrier = training_table(records["carrier"])
        carrier["cell"] = carrier.protocol
        carrier["condition"] = carrier.apply(
            lambda row: "masked" if row.cache_name == "sipakmed_masked"
            else ("colour" if row.color_augment else "standard"), axis=1)
        carrier.to_csv(output / "carrier_runs.csv", index=False)
    if "linadv" in records:
        training_table(records["linadv"], prober).to_csv(output / "linadv_recoverability.csv", index=False)
    if "normalization" in records:
        normalised = training_table(records["normalization"], prober)
        normalised["strength"] = normalised.cache_name.str.extract(r"reinhard(\d+)")[0].astype(float) / 100.0
        phase1 = pd.read_csv(Path(args.study2_analysis) / "runs" / "phase1_runs.csv")
        reference = phase1.loc[phase1.architecture.isin(NORMALIZATION_ARCHITECTURES)
                               & phase1.seed.isin(FOLLOWUP_SPLIT_SEEDS) & phase1.protocol.isin(["P1", "P2"])]
        if len(reference) != len(NORMALIZATION_ARCHITECTURES) * len(FOLLOWUP_SPLIT_SEEDS) * 2:
            raise SystemExit("normalisation reference runs are incomplete")
        base = training_table([json.loads(Path(path).read_text()) for path in reference.run_json], prober,
                              strength=0.0)
        pd.concat([base, normalised], ignore_index=True).to_csv(output / "normalization_recoverability.csv",
                                                               index=False)
    grl_path = Path(args.study2_analysis) / "mechanism" / "grl_recoverability.csv"
    if grl_path.is_file():
        # Secondary: agreement of the device probe with the registered sklearn probe on the H5 runs.
        grl = pd.read_csv(grl_path)
        agreement = []
        for item in grl.itertuples(index=False):
            run = json.loads(Path(item.run_json).read_text())
            torch_value = prober(run["artifacts"]["probe_features"]["path"])["within_class_accuracy"]
            agreement.append({"run_json": item.run_json, "sklearn": item.within_class_accuracy, "device": torch_value})
        pd.DataFrame(agreement).to_csv(output / "probe_agreement.csv", index=False)


if __name__ == "__main__":
    main()
