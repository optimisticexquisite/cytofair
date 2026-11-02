#!/usr/bin/env python3
"""Run the complete resumable Cx22 semantic+instance matrix on at most three GPUs."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SEEDS = (17, 29, 43, 71, 101)
PROTOCOLS = ("P1", "P1-prime", "P2", "P3", "OFFICIAL")
SEMANTIC_ARCHITECTURES = (
    "unet_resnet34",
    "unetplusplus_resnet34",
    "deeplabv3plus_resnet50",
    "fpn_resnet34",
    "manet_resnet34",
)
INSTANCE_ARCHITECTURES = ("maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2")


@dataclass(frozen=True)
class Task:
    task: str
    architecture: str
    protocol: str
    seed: int

    @property
    def stem(self) -> str:
        protocol = self.protocol.lower().replace("-", "_")
        return f"{self.task}_{self.architecture}_{protocol}_seed{self.seed}"


def idle_gpus() -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    available = []
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        if memory <= 512 and utilization <= 5:
            available.append(index)
    return available


def task_grid(selected: list[str]) -> list[Task]:
    tasks = []
    for task in selected:
        architectures = SEMANTIC_ARCHITECTURES if task == "semantic" else INSTANCE_ARCHITECTURES
        # Seed-blocked protocol order yields early within-seed P1/P1'/P2/P3 evidence.
        tasks.extend(
            Task(task, architecture, protocol, seed)
            for architecture in architectures
            for seed in SEEDS
            for protocol in PROTOCOLS
        )
    return tasks


def _last_epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 131_072))
        lines = handle.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        if '"event": "epoch"' not in line:
            continue
        try:
            return int(json.loads(line)["epoch"])
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    return 0


def _write_status(path: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _valid_result(path: Path, task: Task) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_task = f"{task.task}_segmentation"
    return (
        record.get("task") == expected_task
        and record.get("architecture") == task.architecture
        and record.get("protocol") == task.protocol
        and record.get("seed") == task.seed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument("--instance-batch-size", type=int, default=2)
    parser.add_argument("--max-gpus", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--tasks", nargs="+", choices=("semantic", "instance"), default=["semantic", "instance"])
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "matrix_status.json"
    all_tasks = task_grid(args.tasks)
    completed = [task for task in all_tasks if _valid_result(output / f"{task.stem}.json", task)]
    pending = [task for task in all_tasks if task not in completed]
    gpus = idle_gpus()[: args.max_gpus]
    if not gpus:
        raise RuntimeError("No idle GPU satisfies memory<=512 MiB and utilization<=5%")

    attempts = {task: 0 for task in all_tasks}
    running: dict[int, tuple[subprocess.Popen, object, Task, Path]] = {}
    failures: list[dict] = []
    started_at = datetime.now(UTC).isoformat()
    state = {
        "state": "running",
        "controller_pid": os.getpid(),
        "started_at_utc": started_at,
        "map": str(Path(args.map).resolve()),
        "output": str(output),
        "gpus": gpus,
        "epochs": args.epochs,
        "total": len(all_tasks),
        "completed": len(completed),
        "pending": len(pending),
        "active": [],
        "failures": failures,
    }

    def update() -> None:
        active = []
        for gpu, (process, _, task, log) in running.items():
            active.append(
                {
                    **asdict(task),
                    "gpu": gpu,
                    "pid": process.pid,
                    "attempt": attempts[task],
                    "epoch": _last_epoch(log),
                    "log": str(log),
                }
            )
        state.update(
            completed=len(completed),
            pending=len(pending),
            active=sorted(active, key=lambda item: item["gpu"]),
            failures=failures,
        )
        _write_status(status_path, state)

    def stop_children(*_):
        for process, log_handle, _, _ in running.values():
            process.terminate()
            log_handle.close()
        state["state"] = "interrupted"
        update()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    update()
    while pending or running:
        while pending and len(running) < len(gpus) and not failures:
            task = pending.pop(0)
            attempts[task] += 1
            result = output / f"{task.stem}.json"
            log = output / f"{task.stem}.attempt{attempts[task]}.log"
            command = [
                sys.executable,
                "-m",
                "cytofair.cli",
                f"train-{task.task}-segmentation",
                args.map,
                "--protocol",
                task.protocol,
                "--architecture",
                task.architecture,
                "--seed",
                str(task.seed),
                "--output",
                str(result),
                "--epochs",
                str(args.epochs),
                "--workers",
                str(args.workers),
                "--image-size",
                str(args.image_size),
                "--batch-size",
                str(
                    args.semantic_batch_size if task.task == "semantic" else args.instance_batch_size
                ),
                "--repository",
                str(Path.cwd()),
            ]
            gpu = next(index for index in gpus if index not in running)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            log_handle = log.open("a")
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[gpu] = (process, log_handle, task, log)
            print(
                f"started gpu={gpu} attempt={attempts[task]} task={task.stem} pid={process.pid}",
                flush=True,
            )
            update()
        time.sleep(10)
        for gpu, (process, log_handle, task, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del running[gpu]
            result = output / f"{task.stem}.json"
            if return_code == 0 and _valid_result(result, task):
                completed.append(task)
                print(f"finished gpu={gpu} task={task.stem}", flush=True)
            elif attempts[task] < args.max_attempts:
                pending.insert(0, task)
                print(
                    f"retrying gpu={gpu} rc={return_code} task={task.stem} "
                    f"checkpoint_epoch={_last_epoch(log)}",
                    flush=True,
                )
            else:
                failures.append(
                    {
                        **asdict(task),
                        "return_code": return_code,
                        "attempts": attempts[task],
                        "last_epoch": _last_epoch(log),
                        "last_log": str(log),
                    }
                )
                print(f"failed rc={return_code} task={task.stem}", flush=True)
            update()
        if failures:
            for process, log_handle, _, _ in running.values():
                process.terminate()
                log_handle.close()
            running.clear()
            state["state"] = "failed"
            update()
            raise RuntimeError(f"Cx22 matrix stopped after failed cell: {failures[0]}")
        # Refresh epoch counters and controller heartbeat even when no cell has
        # finished; this is the user's primary detached-run health signal.
        update()
    state["state"] = "completed"
    state["completed_at_utc"] = datetime.now(UTC).isoformat()
    update()
    print(f"Cx22 matrix complete: {len(completed)}/{len(all_tasks)}", flush=True)


if __name__ == "__main__":
    main()
