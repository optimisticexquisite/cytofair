"""Build the structural-leakage table from canonical provenance maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import CONFIRMATORY_SEEDS
from .splits import (
    analytic_straddle,
    make_grouped,
    make_p1,
    monte_carlo_straddle_interval,
    structural_leakage,
)


def _format_mean_sd(values: Sequence[float], scale: float = 1.0, digits: int = 2) -> str:
    values = np.asarray(values, dtype=float) * scale
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def structural_table(
    maps: Mapping[str, pd.DataFrame], seeds: Sequence[int] = CONFIRMATORY_SEEDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paper-ready summary and seed-level audit rows."""
    if len(seeds) < 5:
        raise ValueError("Table 1 requires at least five seeds")
    summaries = []
    seed_rows = []
    for dataset, frame in maps.items():
        parent_sizes = frame.groupby("parent_id", observed=True).size().to_numpy()
        analytic = analytic_straddle(parent_sizes)
        mc_low, mc_high = monte_carlo_straddle_interval(parent_sizes)
        p1_straddle, p1_test_leak, p2_straddle, p3_straddle = [], [], [], []
        for seed in seeds:
            p1 = make_p1(frame, seed)
            p2 = make_grouped(frame, seed)
            p3_group = "p3_group_id" if "p3_group_id" in frame.columns else "parent_id"
            p3 = make_grouped(
                frame, seed, protocol="P3", group_column=p3_group
            )
            p1_metrics = structural_leakage(p1)
            p2_metrics = structural_leakage(p2)
            p3_metrics = structural_leakage(p3)
            p1_straddle.append(p1_metrics.straddle_fraction)
            p1_test_leak.append(p1_metrics.leaked_test_crop_fraction)
            p2_straddle.append(p2_metrics.straddle_fraction)
            p3_straddle.append(p3_metrics.straddle_fraction)
            seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "p1_straddle_fraction": p1_metrics.straddle_fraction,
                    "p1_leaked_test_crop_fraction": p1_metrics.leaked_test_crop_fraction,
                    "p2_straddle_fraction": p2_metrics.straddle_fraction,
                    "p3_straddle_fraction": p3_metrics.straddle_fraction,
                }
            )
        observed = float(np.mean(p1_straddle))
        is_control = bool((parent_sizes == 1).all())
        control_pass = bool(not is_control or (max(p1_straddle) == 0 and max(p1_test_leak) == 0))
        summaries.append(
            {
                "dataset": dataset,
                "crops": len(frame),
                "parents": frame.parent_id.nunique(),
                "crops_per_parent_mean_sd": _format_mean_sd(parent_sizes),
                "analytic_p1_non_straddle_pct": (1.0 - analytic) * 100,
                "analytic_p1_straddle_pct": analytic * 100,
                "monte_carlo_95_low_pct": mc_low * 100,
                "monte_carlo_95_high_pct": mc_high * 100,
                "observed_p1_straddle_pct_mean_sd": _format_mean_sd(p1_straddle, 100),
                "observed_p1_test_leak_pct_mean_sd": _format_mean_sd(p1_test_leak, 100),
                "observed_p2_straddle_pct_mean_sd": _format_mean_sd(p2_straddle, 100),
                "observed_p3_straddle_pct_mean_sd": _format_mean_sd(p3_straddle, 100),
                "analytic_check_passed": bool(mc_low <= observed <= mc_high),
                "protocol_ordering_passed": bool(
                    np.mean(p3_straddle) <= np.mean(p2_straddle) <= np.mean(p1_straddle)
                ),
                "negative_control": is_control,
                "negative_control_passed": control_pass,
                "n_seeds": len(seeds),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(seed_rows)
