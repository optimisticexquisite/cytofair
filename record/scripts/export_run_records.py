#!/usr/bin/env python3
"""Export every Study 2 and Amendment 8 run record into compact, committed tables.

The full JSON records (with artifact paths into node-local storage) are archived on
the training host; these tables keep everything the lifecycle record plots: per-run
metrics, timing, GPU, configuration, per-epoch histories, and confusion matrices.

Usage: export_run_records.py RECORD_DATA_DIR OUTPUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TRAINING = ("phase1", "factorial", "grl", "cric", "followup/carrier", "followup/linadv", "followup/normalization")
FIELDS = ("dataset", "protocol", "architecture", "seed", "accuracy", "macro_f1", "balanced_accuracy", "nll", "ece_15",
          "train_crops", "test_crops", "train_parents", "test_parents", "test_crops_with_train_parent_fraction",
          "adversary_lambda", "adversary_kind", "color_augment", "cache_name", "feature_dim", "weights", "git_sha",
          "training_seconds", "elapsed_seconds", "started_at_utc", "completed_at_utc", "gpu_visible_devices",
          "split_file_sha256")


def main() -> None:
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    runs, histories, confusion, versions = [], [], {}, {}
    for experiment in TRAINING:
        for path in sorted((source / "runs" / experiment / "runs").glob("*.json")):
            record = json.loads(path.read_text())
            name = path.stem
            row = {"experiment": experiment.split("/")[-1], "run": name, **{k: record.get(k) for k in FIELDS}}
            runs.append(row)
            for item in record.get("history", []):
                histories.append({"experiment": row["experiment"], "run": name, **item})
            confusion[f"{row['experiment']}/{name}"] = {"classes": record["class_names"],
                                                         "matrix": record["confusion_matrix"]}
            runtime = record.get("runtime", {})
            key = (runtime.get("torch_version"), runtime.get("cuda_version"), runtime.get("gpu_name"))
            versions.setdefault(str(key), {k: v for k, v in runtime.items() if k != "command"})
    evaluations = []
    for experiment in ("factorial_eval", "leace"):
        for path in sorted((source / "runs" / "followup" / experiment / "runs").glob("*.json")):
            record = json.loads(path.read_text())
            evaluations.append({"experiment": experiment, "run": path.stem, "architecture": record["architecture"],
                                "protocol": record["protocol"], "seed": record["seed"],
                                "elapsed_seconds": record["elapsed_seconds"],
                                "completed_at_utc": record["completed_at_utc"], "git_sha": record["git_sha"]})
    smoke = []
    for path in sorted((source / "smoke" / "runs").glob("*.json")) + sorted(
            (source / "smoke_followup").glob("*/runs/*.json")):
        record = json.loads(path.read_text())
        smoke.append({"run": path.stem, "experiment": record.get("experiment", "train"),
                      "architecture": record.get("architecture"), "protocol": record.get("protocol"),
                      "epochs": record.get("epochs"), "elapsed_seconds": record.get("elapsed_seconds")})
    fingerprint = json.loads((source / "runs" / "fingerprint" / "fingerprint_encoder.json").read_text())
    pd.DataFrame(runs).to_csv(output / "training_runs.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "histories.csv", index=False)
    pd.DataFrame(evaluations).to_csv(output / "evaluation_runs.csv", index=False)
    pd.DataFrame(smoke).to_csv(output / "smoke_runs.csv", index=False)
    (output / "confusion_matrices.json").write_text(json.dumps(confusion) + "\n")
    (output / "runtime_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (output / "fingerprint_encoder.json").write_text(json.dumps(
        {k: v for k, v in fingerprint.items() if k not in {"artifacts"}}, indent=2) + "\n")
    for log in ("runs/controller.log", "runs/followup/controller.log"):
        target = output / ("controller_study2.log" if "followup" not in log else "controller_followup.log")
        target.write_text((source / log).read_text())
    print(len(runs), "training runs,", len(histories), "epochs,", len(evaluations), "evaluations,", len(smoke), "smoke")


if __name__ == "__main__":
    main()
