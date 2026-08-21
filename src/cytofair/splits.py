"""Leakage-aware split generation and structural measurements."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass(frozen=True)
class LeakageMetrics:
    n_crops: int
    n_parents: int
    n_test_crops: int
    n_straddled_parents: int
    straddle_fraction: float
    n_leaked_test_crops: int
    leaked_test_crop_fraction: float


def _validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"crop_id", "parent_id", "class_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Split source misses columns: {missing}")
    if not frame.crop_id.is_unique:
        raise ValueError("crop_id must be unique before splitting")
    if frame[list(required)].isna().any().any():
        raise ValueError("split source contains missing IDs or labels")
    return frame.reset_index(drop=True).copy()


def _partition_frame(frame: pd.DataFrame, test_indices: np.ndarray, protocol: str, seed: int):
    out = frame[["crop_id", "parent_id", "class_name"]].copy()
    out["partition"] = "train"
    out.loc[test_indices, "partition"] = "test"
    out["protocol"] = protocol
    out["seed"] = seed
    return out


def make_p1(frame: pd.DataFrame, seed: int, test_size: float = 0.2) -> pd.DataFrame:
    source = _validate_source(frame)
    indices = np.arange(len(source))
    _, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=source.class_name,
    )
    return _partition_frame(source, np.asarray(test), "P1", seed)


def make_grouped(
    frame: pd.DataFrame,
    seed: int,
    *,
    protocol: str = "P2",
    group_column: str = "parent_id",
    n_splits: int = 5,
) -> pd.DataFrame:
    source = _validate_source(frame)
    if group_column not in frame.columns:
        raise ValueError(f"Missing grouping column: {group_column}")
    groups = frame[group_column].astype("string")
    if groups.isna().any():
        raise ValueError(f"Grouping column {group_column} contains missing values")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test = next(splitter.split(source.crop_id, source.class_name, groups))
    out = _partition_frame(source, np.asarray(test), protocol, seed)
    leakage = structural_leakage(out)
    if leakage.n_straddled_parents:
        raise RuntimeError(f"{protocol} produced {leakage.n_straddled_parents} straddled parents")
    return out


def make_p1_prime(p1: pd.DataFrame, p2: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Size-match P1 training data to P2 without changing the P1 test set."""
    if set(p1.crop_id) != set(p2.crop_id):
        raise ValueError("P1 and P2 must cover the same crops")
    target = int((p2.partition == "train").sum())
    train = p1.loc[p1.partition == "train"].copy()
    if target > len(train):
        raise ValueError("P2 training size exceeds P1 training size")
    if target < len(train):
        # A seeded stratified sample. The exact target is guaranteed by train_test_split.
        keep, _ = train_test_split(
            np.arange(len(train)),
            train_size=target,
            random_state=seed,
            shuffle=True,
            stratify=train.class_name,
        )
        retained = set(train.iloc[keep].crop_id)
    else:
        retained = set(train.crop_id)
    out = p1.loc[(p1.partition == "test") | p1.crop_id.isin(retained)].copy()
    out["protocol"] = "P1-prime"
    out["seed"] = seed
    out["included"] = True
    return out.reset_index(drop=True)


def structural_leakage(split: pd.DataFrame) -> LeakageMetrics:
    required = {"crop_id", "parent_id", "partition"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Split misses columns: {missing}")
    if not set(split.partition).issubset({"train", "test"}):
        raise ValueError("partition must contain only train/test")
    by_parent = split.groupby("parent_id", observed=True).partition.agg(set)
    straddled = set(by_parent.index[by_parent.map(lambda values: values == {"train", "test"})])
    test = split.loc[split.partition == "test"]
    leaked_test = int(test.parent_id.isin(straddled).sum())
    return LeakageMetrics(
        n_crops=len(split),
        n_parents=split.parent_id.nunique(),
        n_test_crops=len(test),
        n_straddled_parents=len(straddled),
        straddle_fraction=len(straddled) / len(by_parent) if len(by_parent) else float("nan"),
        n_leaked_test_crops=leaked_test,
        leaked_test_crop_fraction=leaked_test / len(test) if len(test) else float("nan"),
    )


def analytic_straddle(parent_sizes: Iterable[int], train_fraction: float = 0.8) -> float:
    sizes = np.asarray(tuple(parent_sizes), dtype=float)
    if sizes.size == 0 or (sizes <= 0).any():
        raise ValueError("parent sizes must be non-empty positive integers")
    test_fraction = 1.0 - train_fraction
    return float(np.mean(1.0 - np.power(train_fraction, sizes) - np.power(test_fraction, sizes)))


def monte_carlo_straddle_interval(
    parent_sizes: Iterable[int],
    *,
    train_fraction: float = 0.8,
    repetitions: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bernoulli-allocation interval corresponding exactly to the analytic formula."""
    sizes = np.asarray(tuple(parent_sizes), dtype=int)
    if sizes.size == 0 or (sizes <= 0).any():
        raise ValueError("parent sizes must be non-empty positive integers")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        train_counts = rng.binomial(sizes, train_fraction)
        values[index] = np.mean((train_counts > 0) & (train_counts < sizes))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)

