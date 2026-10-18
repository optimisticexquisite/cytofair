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
import re
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


def divergence_epoch(task: dict, log_dir: str | None) -> int | None:
    """Epoch at which every attempt of a run hit a non-finite loss, or None."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", help="registered JSONL task files")
    parser.add_argument("--output", required=True, help="output directory for tidy tables")
    parser.add_argument("--allow-incomplete", action="store_true", help="progress reports only")
    parser.add_argument(
        "--diverged-logs",
        help="attempt-log directory; a missing gradient-reversal run whose attempts ended in a non-finite "
        "loss is recorded as diverged rather than missing (Amendment 7)",
    )
    args = parser.parse_args()

    tasks = []
    for path in args.tasks:
        tasks.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    rows, problems, diverged = [], [], []
    for task in tasks:
        run_path = Path(task["run_json"])
        if not run_path.is_file():
            epoch = divergence_epoch(task, args.diverged_logs)
            if task["experiment"] == "grl" and epoch is not None:
                diverged.append({"task_id": task["id"], **task["expect"], "diverged_at_epoch": epoch})
            else:
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
    pd.DataFrame(
        diverged, columns=["task_id", "dataset", "protocol", "architecture", "seed", "split_file_sha256",
                           "adversary_lambda", "diverged_at_epoch"]
    ).to_csv(output / "diverged.csv", index=False)
    (output / "collection_report.json").write_text(
        json.dumps(
            {"tasks": len(tasks), "collected": len(rows), "diverged": len(diverged), "problems": problems}, indent=2
        ) + "\n"
    )
    print(f"collected {len(rows)}/{len(tasks)} runs; {len(diverged)} diverged; {len(problems)} problems")
    for problem in problems[:20]:
        print(" ", problem)
    if problems and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
