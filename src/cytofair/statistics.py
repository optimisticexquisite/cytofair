"""Preregistered summaries, paired inference, and multiplicity correction."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def mean_sd(
    frame: pd.DataFrame,
    *,
    value: str,
    by: Sequence[str],
    min_seeds: int = 5,
) -> pd.DataFrame:
    summary = (
        frame.groupby(list(by), dropna=False, observed=True)[value]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary["complete"] = summary.n >= min_seeds
    summary.loc[~summary.complete, ["mean", "sd"]] = np.nan
    return summary


def _cohens_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("inf") if np.mean(diff) != 0 else 0.0
    return float(np.mean(diff) / sd)


def _rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def _stable_seed(parts: Sequence[object]) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bootstrap_ci(diff: np.ndarray, estimator, seed: int, repetitions: int = 10_000):
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for idx in range(repetitions):
        estimates[idx] = estimator(rng.choice(diff, size=len(diff), replace=True))
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(finite, [0.025, 0.975]))


def paired_contrasts(
    runs: pd.DataFrame,
    *,
    value: str,
    family_columns: Sequence[str],
    left_protocol: str,
    right_protocol: str,
    min_seeds: int = 5,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Compare left-right per family, then BH-correct the emitted p-values."""
    required = set(family_columns) | {"protocol", "seed", value}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Run table misses columns: {missing}")
    records: list[dict] = []
    group_key = list(family_columns)
    for key, group in runs.groupby(group_key, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        wide = group.pivot_table(index="seed", columns="protocol", values=value, aggfunc="first")
        if left_protocol not in wide or right_protocol not in wide:
            continue
        paired = wide[[left_protocol, right_protocol]].dropna()
        diff = (paired[left_protocol] - paired[right_protocol]).to_numpy(dtype=float)
        record = dict(zip(family_columns, key))
        record.update(
            left_protocol=left_protocol,
            right_protocol=right_protocol,
            n=len(diff),
            mean_difference=float(np.mean(diff)) if len(diff) else np.nan,
            sd_difference=float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
            test="incomplete",
            statistic=np.nan,
            p_value=np.nan,
            effect_name="",
            effect_size=np.nan,
            effect_ci_low=np.nan,
            effect_ci_high=np.nan,
        )
        if len(diff) >= min_seeds:
            if np.all(diff == 0):
                record.update(test="all_zero_descriptive", effect_name="cohens_dz", effect_size=0.0)
            elif np.ptp(diff) == 0:
                direction = float(np.sign(diff[0]))
                infinite_effect = float(np.copysign(np.inf, direction))
                record.update(
                    test="constant_nonzero_difference",
                    statistic=infinite_effect,
                    p_value=0.0,
                    effect_name="cohens_dz",
                    effect_size=infinite_effect,
                    effect_ci_low=infinite_effect,
                    effect_ci_high=infinite_effect,
                    shapiro_p=np.nan,
                )
            else:
                shapiro_p = float(stats.shapiro(diff).pvalue)
                record["shapiro_p"] = shapiro_p
                if shapiro_p >= 0.05:
                    result = stats.ttest_rel(
                        paired[left_protocol], paired[right_protocol], alternative="two-sided"
                    )
                    estimator = _cohens_dz
                    record.update(
                        test="paired_t",
                        statistic=float(result.statistic),
                        p_value=float(result.pvalue),
                        effect_name="cohens_dz",
                        effect_size=estimator(diff),
                    )
                else:
                    result = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                    estimator = _rank_biserial
                    record.update(
                        test="wilcoxon",
                        statistic=float(result.statistic),
                        p_value=float(result.pvalue),
                        effect_name="rank_biserial",
                        effect_size=estimator(diff),
                    )
                ci = _bootstrap_ci(
                    diff,
                    estimator,
                    _stable_seed((*key, left_protocol, right_protocol)),
                    repetitions=bootstrap_repetitions,
                )
                record.update(effect_ci_low=ci[0], effect_ci_high=ci[1])
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["q_value"] = np.nan
    eligible = out.p_value.notna()
    if eligible.any():
        out.loc[eligible, "q_value"] = multipletests(
            out.loc[eligible, "p_value"], method="fdr_bh"
        )[1]
    return out


def verify_protocol_order(
    runs: pd.DataFrame,
    *,
    value: str,
    by: Sequence[str],
    order: Sequence[str] = ("P3", "P2", "P1"),
) -> pd.DataFrame:
    means = runs.groupby([*by, "protocol"], observed=True)[value].mean().unstack("protocol")
    for protocol in order:
        if protocol not in means:
            means[protocol] = np.nan
    pairs = list(pairwise(order))
    means["ordering_passed"] = np.logical_and.reduce(
        [(means[left] <= means[right]) for left, right in pairs]
    )
    return means.reset_index()


def shuffled_probe_check(
    shuffled_scores: Sequence[float], chance: float, confidence: float = 0.95
) -> dict[str, float | bool]:
    scores = np.asarray(shuffled_scores, dtype=float)
    if scores.size < 5:
        raise ValueError("At least five shuffled-label seeds are required")
    alpha = 1.0 - confidence
    sem = stats.sem(scores)
    if sem == 0:
        interval = (float(scores.mean()), float(scores.mean()))
    else:
        interval = stats.t.interval(
            confidence, df=len(scores) - 1, loc=scores.mean(), scale=sem
        )
    return {
        "n": len(scores),
        "mean": float(scores.mean()),
        "sd": float(scores.std(ddof=1)),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "chance": float(chance),
        "passed": bool(interval[0] <= chance <= interval[1]),
        "alpha": float(alpha),
    }
