#!/usr/bin/env python3
"""Run the preregistered probe matrix on at most three demonstrably idle GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (17, 29, 43, 71, 101)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-gpus", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument(
        "--modes", nargs="+", choices=("rgb", "grayscale", "highpass"), default=("rgb",)
    )
    parser.add_argument("--without-shuffled", action="store_true")
    args = parser.parse_args()
    gpus = idle_gpus()[: args.max_gpus]
    if not gpus:
        raise RuntimeError("No idle GPU satisfies memory<=512 MiB and utilization<=5%")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for mode in args.modes:
        for seed in SEEDS:
            tasks.append((mode, seed, False))
        if not args.without_shuffled and mode == "rgb":
            for seed in SEEDS:
                tasks.append((mode, seed, True))

    running = {}
    failures = []
    while tasks or running:
        while tasks and len(running) < len(gpus):
            mode, seed, shuffled = tasks.pop(0)
            suffix = "shuffled" if shuffled else "observed"
            result = output / f"probe_{mode}_{suffix}_seed{seed}.json"
            if result.exists():
                continue
            gpu = next(index for index in gpus if index not in running)
            log = output / f"probe_{mode}_{suffix}_seed{seed}.log"
            command = [
                sys.executable,
                "-m",
                "cytofair.cli",
                "train-probe",
                args.map,
                "--seed",
                str(seed),
                "--output",
                str(result),
                "--image-mode",
                mode,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ]
            if shuffled:
                command.append("--shuffled-labels")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_handle = log.open("w")
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, log_handle, result)
            print(f"started gpu={gpu} result={result}", flush=True)
        time.sleep(5)
        for gpu, (process, log_handle, result) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            print(f"finished gpu={gpu} rc={return_code} result={result}", flush=True)
            if return_code:
                failures.append(str(result))
            del running[gpu]
    if failures:
        raise RuntimeError(f"Probe runs failed: {failures}")


if __name__ == "__main__":
    main()

