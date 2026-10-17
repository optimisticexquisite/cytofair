#!/usr/bin/env python3
"""Collect Study 2 run JSONs into tidy tables and verify registered completeness.

Every registered task (from the JSONL task files) must have exactly one valid
result whose split hash equals the registered split manifest. Missing,
duplicated, or mismatched runs are listed and make the command fail, so no
summary can silently drop a cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FIELDS = (
    "dataset",
    "protocol",
    "architecture",
    "seed",
    "adversary_lambda",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "nll",
    "ece_15",
    "train_crops",
    "test_crops",
    "train_parents",
    "test_parents",
    "test_crops_with_train_parent_fraction",
    "split_file_sha256",
    "split_sha256",
    "git_sha",
    "training_seconds",
    "elapsed_seconds",
    "gpu_visible_devices",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", help="registered JSONL task files")
    parser.add_argument("--output", required=True, help="output directory for tidy tables")
    parser.add_argument("--allow-incomplete", action="store_true", help="progress reports only")
    args = parser.parse_args()

    tasks = []
    for path in args.tasks:
        tasks.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    rows, problems = [], []
    for task in tasks:
        run_path = Path(task["run_json"])
        if not run_path.is_file():
            problems.append(f"missing: {task['id']}")
            continue
        record = json.loads(run_path.read_text())
        expected = task["expect"]
        for key, value in expected.items():
            if record.get(key) != value:
                problems.append(f"{task['id']}: {key}={record.get(key)!r} expected {value!r}")
        row = {field: record.get(field) for field in FIELDS}
        row.update(task_id=task["id"], experiment=task["experiment"], run_json=str(run_path))
        rows.append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        duplicated = frame.duplicated("task_id")
        problems.extend(f"duplicate: {task_id}" for task_id in frame.task_id[duplicated])
        for experiment, group in frame.groupby("experiment"):
            group.drop(columns="experiment").to_csv(output / f"{experiment}_runs.csv", index=False)
    (output / "collection_report.json").write_text(
        json.dumps({"tasks": len(tasks), "collected": len(rows), "problems": problems}, indent=2) + "\n"
    )
    print(f"collected {len(rows)}/{len(tasks)} runs; {len(problems)} problems")
    for problem in problems[:20]:
        print(" ", problem)
    if problems and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
