"""Structural leakage-risk report computable from a crop-to-parent table alone.

Everything here follows from the parent-size distribution and the class labels,
under independent crop-level allocation with test fraction ``q``:

* expected parent straddle: mean over parents of ``1 - (1-q)^k - q^k``;
* expected fraction of test crops with a training sibling: size-biased mean of
  ``1 - q^(k-1)``;
* expected number of training siblings per test crop: size-biased mean of
  ``(1-q)(k-1)``;
* diversity gap: expected distinct training parents under crop-level splitting,
  ``sum(1 - q^k)``, versus ``(1-q) * parents`` under parent-level splitting;
* class-parent dependence: the share of class entropy explained by parent
  identity, ``I(class; parent) / H(class)`` (1 when every parent is single-class).

These are the ingredients that make a crop-level split optimistic; they do not
by themselves predict the size of the accuracy inflation, which also depends on
how learnable the parent fingerprint is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def structural_risk(provenance: pd.DataFrame, *, test_fraction: float = 0.2) -> dict:
    required = {"crop_id", "parent_id", "class_name"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Risk report needs columns: {missing}")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    q = float(test_fraction)
    sizes = provenance.groupby("parent_id").size().to_numpy(dtype=float)
    crops = sizes.sum()
    straddle = float(np.mean(np.clip(1 - (1 - q) ** sizes - q**sizes, 0.0, 1.0)))
    test_leak = float((sizes * (1 - q ** (sizes - 1))).sum() / crops)
    siblings = float((sizes * (1 - q) * (sizes - 1)).sum() / crops)
    crop_level_parents = float((1 - q**sizes).sum())
    parent_level_parents = (1 - q) * len(sizes)
    shares = sizes / crops

    class_marginal = provenance.class_name.value_counts(normalize=True).to_numpy()
    h_class = _entropy(class_marginal)
    h_conditional = 0.0
    purity = []
    for _, group in provenance.groupby("parent_id"):
        distribution = group.class_name.value_counts(normalize=True).to_numpy()
        h_conditional += len(group) / crops * _entropy(distribution)
        purity.append(distribution.max())
    mutual = h_class - h_conditional
    return {
        "crops": int(crops),
        "parents": len(sizes),
        "crops_per_parent_mean": float(sizes.mean()),
        "singleton_parent_fraction": float((sizes == 1).mean()),
        "effective_parent_number": float(np.exp(-(shares * np.log(shares)).sum())),
        "test_fraction": q,
        "expected_parent_straddle": straddle,
        "expected_test_crops_with_training_sibling": test_leak,
        "expected_training_siblings_per_test_crop": siblings,
        "expected_training_parents_crop_split": crop_level_parents,
        "expected_training_parents_parent_split": parent_level_parents,
        "training_diversity_ratio": crop_level_parents / parent_level_parents,
        "class_entropy_bits": h_class,
        "class_parent_mutual_information_bits": mutual,
        "class_parent_dependence": mutual / h_class if h_class > 0 else float("nan"),
        "mean_parent_majority_class_purity": float(np.mean(purity)),
    }
