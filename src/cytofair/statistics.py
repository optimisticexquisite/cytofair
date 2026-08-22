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
    apply_bh: bool = True,
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
    if apply_bh and eligible.any():
        out.loc[eligible, "q_value"] = multipletests(
            out.loc[eligible, "p_value"], method="fdr_bh"
        )[1]
    return out


def apply_bh_family(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Apply BH once over exactly the supplied, explicitly stated family."""
    if "p_value" not in contrasts.columns:
        raise ValueError("Contrast family has no p_value column")
    out = contrasts.copy()
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


def ranking_stability(
    runs: pd.DataFrame,
    *,
    value: str,
    dataset_column: str = "dataset",
    architecture_column: str = "architecture",
    left_protocol: str = "P1",
    right_protocol: str = "P2",
    min_seeds: int = 5,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Kendall ranking stability with a seed-level paired bootstrap interval."""
    required = {
        dataset_column,
        architecture_column,
        "protocol",
        "seed",
        value,
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Run table misses ranking columns: {missing}")
    records = []
    for dataset, group in runs.groupby(dataset_column, dropna=False, observed=True):
        selected = group.loc[group.protocol.isin([left_protocol, right_protocol])]
        wide = selected.pivot_table(
            index=architecture_column, columns=["protocol", "seed"], values=value, aggfunc="first"
        )
        left_seeds = set(wide[left_protocol].columns) if left_protocol in wide else set()
        right_seeds = set(wide[right_protocol].columns) if right_protocol in wide else set()
        common_seeds = sorted(left_seeds & right_seeds)
        complete_architectures = [
            architecture
            for architecture in wide.index
            if common_seeds
            and wide.loc[architecture, (left_protocol, common_seeds)].notna().all()
            and wide.loc[architecture, (right_protocol, common_seeds)].notna().all()
        ]
        if len(common_seeds) < min_seeds or len(complete_architectures) < 2:
            continue
        left = wide.loc[complete_architectures, (left_protocol, common_seeds)].to_numpy(
            dtype=float
        )
        right = wide.loc[complete_architectures, (right_protocol, common_seeds)].to_numpy(
            dtype=float
        )
        left_means, right_means = left.mean(axis=1), right.mean(axis=1)
        observed = stats.kendalltau(left_means, right_means)
        rng = np.random.default_rng(
            _stable_seed((dataset, left_protocol, right_protocol, "ranking"))
        )
        bootstrap = np.empty(bootstrap_repetitions, dtype=float)
        for repetition in range(bootstrap_repetitions):
            sample = rng.integers(0, len(common_seeds), size=len(common_seeds))
            bootstrap[repetition] = stats.kendalltau(
                left[:, sample].mean(axis=1), right[:, sample].mean(axis=1)
            ).statistic
        finite = bootstrap[np.isfinite(bootstrap)]
        ci_low, ci_high = (
            tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))
            if len(finite)
            else (float("nan"), float("nan"))
        )
        reordered_pairs = 0
        tie_changes = 0
        for first, second in pairwise_combinations(range(len(complete_architectures))):
            left_sign = np.sign(left_means[first] - left_means[second])
            right_sign = np.sign(right_means[first] - right_means[second])
            reordered_pairs += int(left_sign * right_sign < 0)
            tie_changes += int(left_sign != right_sign and left_sign * right_sign == 0)
        n_architectures = len(complete_architectures)
        records.append(
            {
                dataset_column: dataset,
                "left_protocol": left_protocol,
                "right_protocol": right_protocol,
                "n_architectures": n_architectures,
                "n_seeds": len(common_seeds),
                "test": "kendall_tau_secondary_unadjusted",
                "p_value": float(observed.pvalue),
                "effect_name": "kendall_tau",
                "effect_size": float(observed.statistic),
                "effect_ci_low": ci_low,
                "effect_ci_high": ci_high,
                "reordered_pairs": reordered_pairs,
                "total_pairs": n_architectures * (n_architectures - 1) // 2,
                "tie_changes": tie_changes,
            }
        )
    return pd.DataFrame(records)


def pairwise_combinations(values: Sequence[int]):
    """Yield unordered pairs without importing a second itertools helper at call sites."""
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            yield left, right
