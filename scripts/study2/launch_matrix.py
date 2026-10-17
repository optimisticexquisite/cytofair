#!/usr/bin/env python3
"""Resumable Study 2 GPU scheduler (never more than four GPUs).

Tasks are JSON lines with ``id``, ``priority``, ``gpu_units`` (1 or 2 of each GPU's
two units), ``command`` (argument list), and ``run_json`` (the result file that
proves completion). A task whose result JSON already exists and lists only
existing artifacts is skipped. Failed tasks are retried once; a task that fails
twice is recorded and the remaining matrix continues, because the analysis
itself refuses to summarise an incomplete registered cell.

Before every launch the scheduler checks free space on the output filesystem
(the training host's ``/tmp`` is shared with the operating system) and the
memory already in use on the target GPU (other users may start jobs there).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MAX_GPUS = 4
UNITS_PER_GPU = 2


def gpu_memory_used() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = {}
    for line in output.splitlines():
        index, memory = (int(value.strip()) for value in line.split(","))
        used[index] = memory
    return used


def valid_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    artifacts = record.get("artifacts", {})
    return all(Path(item["path"]).is_file() or item.get("archived") for item in artifacts.values())


def write_status(path: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", help="JSONL task files")
    parser.add_argument("--gpus", required=True, help="comma-separated GPU indices (at most four)")
    parser.add_argument("--logs", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--free-path", default="/tmp")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--gpu-memory-limit-mib", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) > MAX_GPUS or len(set(gpus)) != len(gpus):
        raise SystemExit(f"--gpus must name between one and {MAX_GPUS} distinct devices")
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status)

    tasks = []
    for task_file in args.tasks:
        for line in Path(task_file).read_text().splitlines():
            if line.strip():
                tasks.append(json.loads(line))
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise SystemExit("task identifiers must be unique")
    order = {task["id"]: position for position, task in enumerate(tasks)}
    pending = sorted(
        (task for task in tasks if not valid_result(Path(task["run_json"]))),
        key=lambda task: (task.get("priority", 0), order[task["id"]]),
    )
    completed = len(tasks) - len(pending)
    attempts = {task["id"]: 0 for task in tasks}
    running: dict[str, dict] = {}
    failures: list[dict] = []
    environment_base = os.environ.copy()
    environment_base.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1"
    )
    state = {
        "state": "running",
        "controller_pid": os.getpid(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "gpus": gpus,
        "total": len(tasks),
    }

    def units_in_use(gpu: int) -> int:
        return sum(item["units"] for item in running.values() if item["gpu"] == gpu)

    def update() -> None:
        state.update(
            completed=completed,
            pending=len(pending),
            running=[
                {"id": key, "gpu": item["gpu"], "pid": item["process"].pid, "attempt": attempts[key],
                 "log": str(item["log"]), "started": item["started"]}
                for key, item in sorted(running.items())
            ],
            failures=failures,
            free_gb=shutil.disk_usage(args.free_path).free / 1e9,
        )
        write_status(status_path, state)

    def stop(*_):
        for item in running.values():
            item["process"].terminate()
        state["state"] = "interrupted"
        update()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    update()
    while pending or running:
        launched = True
        while launched and pending:
            launched = False
            if shutil.disk_usage(args.free_path).free / 1e9 < args.min_free_gb:
                state["waiting_for"] = "disk"
                break
            memory = gpu_memory_used()
            for position, task in enumerate(pending):
                units = int(task.get("gpu_units", 1))
                choices = [
                    gpu for gpu in gpus
                    if units_in_use(gpu) + units <= UNITS_PER_GPU
                    and memory.get(gpu, 10**9) <= args.gpu_memory_limit_mib
                ]
                if not choices:
                    continue
                gpu = min(choices, key=units_in_use)
                pending.pop(position)
                attempts[task["id"]] += 1
                log = logs / f"{task['id']}.attempt{attempts[task['id']]}.log"
                environment = dict(environment_base, CUDA_VISIBLE_DEVICES=str(gpu))
                handle = log.open("w")
                process = subprocess.Popen(
                    task["command"], env=environment, stdout=handle, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
                running[task["id"]] = {
                    "task": task, "gpu": gpu, "units": units, "process": process, "handle": handle,
                    "log": log, "started": datetime.now(UTC).isoformat(),
                }
                print(f"started {task['id']} gpu={gpu} attempt={attempts[task['id']]}", flush=True)
                launched = True
                state.pop("waiting_for", None)
                time.sleep(2.0)  # stagger process start-up and cache verification
                break
        time.sleep(args.poll_seconds)
        for key, item in list(running.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["handle"].close()
            del running[key]
            task = item["task"]
            if code == 0 and valid_result(Path(task["run_json"])):
                completed += 1
                print(f"finished {key}", flush=True)
            elif attempts[key] < args.max_attempts:
                pending.insert(0, task)
                print(f"retrying {key} rc={code}", flush=True)
            else:
                failures.append({"id": key, "return_code": code, "attempts": attempts[key], "log": str(item["log"])})
                print(f"failed {key} rc={code}", flush=True)
        update()
    state["state"] = "completed" if not failures else "completed_with_failures"
    state["completed_at_utc"] = datetime.now(UTC).isoformat()
    update()
    print(f"matrix done: {completed}/{len(tasks)} complete, {len(failures)} failed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
