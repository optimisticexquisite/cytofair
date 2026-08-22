"""Paper-ready summaries for preregistered model runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .statistics import apply_bh_family, mean_sd, paired_contrasts, verify_protocol_order


def _semantic_split_hash(split: pd.DataFrame) -> str:
    required = {"crop_id", "partition"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Split misses semantic-hash columns: {missing}")
    payload = split.sort_values("crop_id")[["crop_id", "partition"]].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_classification_matrix(
    result_paths: Sequence[str | Path],
    *,
    split_root: str | Path,
    dataset: str,
    architectures: Sequence[str],
    protocols: Sequence[str],
    seeds: Sequence[int],
    expected_git_sha: str,
    expected_runtime: Mapping[str, object],
    expected_train_crops: int,
    expected_test_crops: int,
) -> dict[str, object]:
    """Fail-closed audit of result keys, environments, and released split hashes."""
    errors: list[str] = []
    records: dict[tuple[str, str, int], dict[str, object]] = {}
    for source in result_paths:
        path = Path(source)
        try:
            record = json.loads(path.read_text())
            key = (str(record["architecture"]), str(record["protocol"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: unreadable experiment key ({exc})")
            continue
        if key in records:
            errors.append(f"duplicate experiment key: {key}")
            continue
        records[key] = record

    expected_keys = {
        (architecture, protocol, int(seed))
        for architecture in architectures
        for protocol in protocols
        for seed in seeds
    }
    missing_keys = sorted(expected_keys - set(records))
    extra_keys = sorted(set(records) - expected_keys)
    if missing_keys:
        errors.append(f"missing experiment keys: {missing_keys}")
    if extra_keys:
        errors.append(f"unexpected experiment keys: {extra_keys}")

    split_hashes: dict[tuple[str, int], str] = {}
    split_root = Path(split_root)
    for protocol in protocols:
        safe_protocol = protocol.lower().replace("-", "_")
        for seed in seeds:
            split_path = split_root / f"{dataset.lower()}_{safe_protocol}_seed{seed}.csv"
            if not split_path.is_file():
                errors.append(f"missing split file: {split_path}")
                continue
            split = pd.read_csv(split_path)
            train = int((split.partition == "train").sum())
            test = int((split.partition == "test").sum())
            if (train, test) != (expected_train_crops, expected_test_crops):
                errors.append(
                    f"{split_path}: split counts {(train, test)} != "
                    f"{(expected_train_crops, expected_test_crops)}"
                )
            split_hashes[(protocol, int(seed))] = _semantic_split_hash(split)

    if "P1" in protocols and "P1-prime" in protocols:
        for seed in seeds:
            left = split_hashes.get(("P1", int(seed)))
            right = split_hashes.get(("P1-prime", int(seed)))
            if left is not None and right is not None and left != right:
                errors.append(f"seed {seed}: P1-prime is not identical to P1")

    provenance_hashes: set[str] = set()
    gpu_ids: set[str] = set()
    for key, record in records.items():
        prefix = f"{key}"
        if record.get("dataset") != dataset or record.get("task") != "classification":
            errors.append(f"{prefix}: wrong dataset/task")
        if record.get("git_sha") != expected_git_sha:
            errors.append(f"{prefix}: git_sha={record.get('git_sha')!r}")
        if (record.get("train_crops"), record.get("test_crops")) != (
            expected_train_crops,
            expected_test_crops,
        ):
            errors.append(f"{prefix}: wrong recorded train/test counts")
        expected_hash = split_hashes.get((key[1], key[2]))
        if expected_hash is not None and record.get("split_sha256") != expected_hash:
            errors.append(f"{prefix}: split hash does not match released split")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            try:
                value = float(record[metric])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{prefix}: invalid {metric}")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"{prefix}: {metric} outside [0, 1]")
        runtime = record.get("runtime")
        if not isinstance(runtime, dict):
            errors.append(f"{prefix}: runtime metadata absent")
            continue
        for name, expected in expected_runtime.items():
            if runtime.get(name) != expected:
                errors.append(f"{prefix}: runtime {name}={runtime.get(name)!r}, expected {expected!r}")
        provenance_hashes.add(str(runtime.get("input_provenance_sha256")))
        gpu_ids.add(str(record.get("gpu_visible_devices")))

    if len(provenance_hashes) != 1:
        errors.append(f"input provenance hashes differ: {sorted(provenance_hashes)}")
    return {
        "passed": not errors,
        "valid_result_files": len(records),
        "expected_result_files": len(expected_keys),
        "dataset": dataset,
        "architectures": list(architectures),
        "protocols": list(protocols),
        "seeds": [int(seed) for seed in seeds],
        "expected_git_sha": expected_git_sha,
        "expected_runtime": dict(expected_runtime),
        "expected_train_crops": expected_train_crops,
        "expected_test_crops": expected_test_crops,
        "input_provenance_sha256": next(iter(provenance_hashes), None)
        if len(provenance_hashes) == 1
        else None,
        "gpu_visible_devices": sorted(gpu_ids),
        "errors": errors,
    }


def read_json_runs(paths: list[str | Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        record["result_path"] = str(Path(path).resolve())
        records.append(record)
    if not records:
        raise ValueError("No run results supplied")
    frame = pd.DataFrame(records)
    keys = ["dataset", "architecture", "protocol", "seed"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Run results contain missing or duplicate experiment keys")
    return frame


def summarize_classification(
    paths: list[str | Path], *, bootstrap_repetitions: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mean/SD, the single confirmatory BH family, secondary P2/P3, and order."""
    runs = read_json_runs(paths)
    required = {"accuracy", "macro_f1", "balanced_accuracy"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Classification results miss metrics: {missing}")
    summaries = []
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        summary = mean_sd(
            runs,
            value=metric,
            by=["dataset", "architecture", "protocol"],
        )
        summary.insert(3, "metric", metric)
        summary["mean_sd"] = summary.apply(
            lambda row: f"{row['mean']:.4f} ± {row['sd']:.4f}" if row.complete else "incomplete",
            axis=1,
        )
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)

    primary_parts = []
    for left in ("P1", "P1-prime"):
        part = paired_contrasts(
            runs,
            value="accuracy",
            family_columns=["dataset", "architecture"],
            left_protocol=left,
            right_protocol="P2",
            bootstrap_repetitions=bootstrap_repetitions,
            apply_bh=False,
        )
        primary_parts.append(part)
    primary = apply_bh_family(pd.concat(primary_parts, ignore_index=True))
    primary["multiplicity_family"] = "classification:P1-v-P2_and_P1-prime-v-P2"

    secondary = paired_contrasts(
        runs,
        value="accuracy",
        family_columns=["dataset", "architecture"],
        left_protocol="P2",
        right_protocol="P3",
        bootstrap_repetitions=bootstrap_repetitions,
        apply_bh=False,
    ).drop(columns="q_value")
    ordering = verify_protocol_order(
        runs, value="accuracy", by=["dataset", "architecture"]
    )
    return summary, primary, secondary, ordering
