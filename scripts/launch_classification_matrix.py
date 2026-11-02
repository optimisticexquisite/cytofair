#!/usr/bin/env python3
"""Run matched classification cells on at most three demonstrably idle GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (17, 29, 43, 71, 101)
ALL_PROTOCOLS = ("P1", "P1-prime", "P2", "P3")
ALL_ARCHITECTURES = (
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v2",
    "vit_b_16",
)


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
    parser.add_argument(
        "--dataset-map",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="repeat for every dataset",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-gpus", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument(
        "--protocols", nargs="+", choices=ALL_PROTOCOLS, default=ALL_PROTOCOLS
    )
    parser.add_argument(
        "--architectures", nargs="+", choices=ALL_ARCHITECTURES, default=("resnet18",)
    )
    args = parser.parse_args()
    gpus = idle_gpus()[: args.max_gpus]
    if not gpus:
        raise RuntimeError("No idle GPU satisfies memory<=512 MiB and utilization<=5%")
    datasets = []
    for specification in args.dataset_map:
        if "=" not in specification:
            raise ValueError(f"Invalid DATASET=PATH specification: {specification}")
        dataset, path = specification.split("=", 1)
        datasets.append((dataset, path))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (dataset, path, architecture, protocol, seed)
        for dataset, path in datasets
        for architecture in args.architectures
        for protocol in args.protocols
        for seed in SEEDS
    ]

    running: dict[int, tuple[subprocess.Popen, object, Path]] = {}
    failures = []
    while tasks or running:
        while tasks and len(running) < len(gpus):
            dataset, path, architecture, protocol, seed = tasks.pop(0)
            safe_dataset = dataset.lower().replace(" ", "_")
            safe_protocol = protocol.lower().replace("-", "_")
            result = output / f"{safe_dataset}_{architecture}_{safe_protocol}_seed{seed}.json"
            if result.exists():
                continue
            gpu = next(index for index in gpus if index not in running)
            log = result.with_suffix(".log")
            command = [
                sys.executable,
                "-m",
                "cytofair.cli",
                "train-classifier",
                path,
                "--dataset",
                dataset,
                "--protocol",
                protocol,
                "--architecture",
                architecture,
                "--seed",
                str(seed),
                "--output",
                str(result),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ]
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
        raise RuntimeError(f"Classification runs failed: {failures}")


if __name__ == "__main__":
    main()
