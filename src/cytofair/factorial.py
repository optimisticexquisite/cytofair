"""Leakage-by-diversity factorial with one common test set per seed (Study 2, arm 2C).

Natural crop-level splitting changes two things at once: test parents appear in
training (leakage) *and* training covers more distinct parents than a
parent-disjoint split of the same size (diversity). This design separates them.

For every seed and class:

* ``E`` (evaluation parents, at least two crops each) supply the common test set:
  ``floor(k/2)`` crops per parent are test crops and the rest are *donor*
  siblings.
* ``U`` parents are reserved as unseen background donors for the counterfactual
  transplant arm and never enter any training or test set.
* ``B`` parents are the clean training pool.

Four training sets share the class-wise crop budget and the test set:

* ``clean_high`` uses every ``B`` parent;
* ``leak_high`` swaps the last ``|E|`` parents of one fixed ``B`` permutation for
  the ``E`` parents' donor siblings;
* ``clean_low`` and ``leak_low`` repeat the construction on the first ``P_low``
  parents of the same permutation.

Within a diversity level, leak and clean therefore differ only in *which* parents
are present (test parents' siblings versus other parents), never in crop count,
parent count, class balance, or allocation rule. Within a leakage level, high and
low differ only in parent count at a fixed crop budget, and the low parent set is
nested inside the high one.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

CELLS = ("leak_high", "leak_low", "clean_high", "clean_low")


@dataclass(frozen=True)
class FactorialConfig:
    evaluation_fraction: float = 0.30
    donor_parents_per_class: int = 6
    donor_parent_min_crops: int = 3
    crops_per_class: int = 260
    low_to_high_ratio: float = 0.70


def _rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, ("cytofair-2C", *parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _even_allocation(capacity: dict[str, int], order: list[str], total: int) -> dict[str, int]:
    """Give every parent one crop, then fill round-robin in ``order`` up to capacity."""
    if total < len(order):
        raise ValueError(f"crop budget {total} is below the parent count {len(order)}")
    if sum(capacity[parent] for parent in order) < total:
        raise ValueError(f"crop budget {total} exceeds parent capacity")
    allocation = {parent: 1 for parent in order}
    remaining = total - len(order)
    while remaining:
        progressed = False
        for parent in order:
            if remaining and allocation[parent] < capacity[parent]:
                allocation[parent] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # pragma: no cover - excluded by the capacity check
            raise RuntimeError("allocation stalled")
    return allocation


DEFAULT_CONFIG = FactorialConfig()


def build_factorial(
    provenance: pd.DataFrame, seed: int, config: FactorialConfig = DEFAULT_CONFIG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(design, cell_summary)``.

    ``design`` has one row per crop with its ``role`` and, for each cell, the
    partition (``train``/``test``/``unused``). Every invariant is checked before
    returning; an infeasible configuration raises instead of silently degrading.
    """
    required = {"crop_id", "parent_id", "class_name"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Factorial provenance misses: {missing}")
    source = provenance[["crop_id", "parent_id", "class_name"]].copy()
    source = source.sort_values("crop_id").reset_index(drop=True)
    if not source.crop_id.is_unique:
        raise ValueError("crop_id must be unique")
    if (source.groupby("parent_id").class_name.nunique() > 1).any():
        raise ValueError("The factorial design assumes one class per parent")

    crops_by_parent = {
        parent: sorted(group.crop_id) for parent, group in source.groupby("parent_id", sort=True)
    }
    parent_class = source.drop_duplicates("parent_id").set_index("parent_id").class_name
    role = {}
    partitions = {cell: {} for cell in CELLS}
    summaries = []
    for class_name in sorted(parent_class.unique()):
        rng = _rng(seed, class_name)
        parents = sorted(parent_class.index[parent_class == class_name])
        sizes = {parent: len(crops_by_parent[parent]) for parent in parents}
        eligible = [parent for parent in parents if sizes[parent] >= 2]
        eligible = [eligible[i] for i in rng.permutation(len(eligible))]
        n_eval = round(config.evaluation_fraction * len(eligible))
        evaluation = eligible[:n_eval]
        donor_pool = [p for p in eligible[n_eval:] if sizes[p] >= config.donor_parent_min_crops]
        unseen = donor_pool[: config.donor_parents_per_class]
        if len(unseen) < config.donor_parents_per_class:
            raise ValueError(f"{class_name}: only {len(unseen)} unseen background parents")
        blocked = set(evaluation) | set(unseen)
        clean_pool = [parent for parent in parents if parent not in blocked]
        clean_pool = [clean_pool[i] for i in rng.permutation(len(clean_pool))]

        test_crops, donor_crops = {}, {}
        for parent in evaluation:
            crops = crops_by_parent[parent]
            order = [crops[i] for i in rng.permutation(len(crops))]
            n_test = len(crops) // 2
            test_crops[parent] = order[:n_test]
            donor_crops[parent] = order[n_test:]
        for parent in evaluation:
            for crop in test_crops[parent]:
                role[crop] = "test"
            for crop in donor_crops[parent]:
                role[crop] = "donor_sibling"
        for parent in unseen:
            for crop in crops_by_parent[parent]:
                role[crop] = "unseen_background"
        for parent in clean_pool:
            for crop in crops_by_parent[parent]:
                role[crop] = "clean_pool"

        n_high = len(clean_pool)
        n_low = round(config.low_to_high_ratio * n_high)
        if n_low < len(evaluation):
            raise ValueError(f"{class_name}: low parent count {n_low} < |E| = {len(evaluation)}")
        parent_sets = {
            "clean_high": clean_pool,
            "clean_low": clean_pool[:n_low],
            "leak_high": evaluation + clean_pool[: n_high - len(evaluation)],
            "leak_low": evaluation + clean_pool[: n_low - len(evaluation)],
        }
        for cell, chosen in parent_sets.items():
            capacity = {
                parent: len(donor_crops[parent]) if parent in donor_crops else sizes[parent]
                for parent in chosen
            }
            allocation_order = [chosen[i] for i in _rng(seed, class_name, cell).permutation(len(chosen))]
            allocation = _even_allocation(capacity, allocation_order, config.crops_per_class)
            for parent, count in allocation.items():
                pool = donor_crops[parent] if parent in donor_crops else crops_by_parent[parent]
                pick = _rng(seed, class_name, cell, parent).permutation(len(pool))[:count]
                for index in pick:
                    partitions[cell][pool[index]] = "train"
            summaries.append(
                {
                    "seed": seed,
                    "class_name": class_name,
                    "cell": cell,
                    "train_parents": len(chosen),
                    "train_crops": config.crops_per_class,
                    "evaluation_parents": len(evaluation),
                    "test_crops": sum(len(v) for v in test_crops.values()),
                    "unseen_background_parents": len(unseen),
                    "clean_pool_parents": n_high,
                }
            )
        for cell in CELLS:
            for parent in evaluation:
                for crop in test_crops[parent]:
                    partitions[cell][crop] = "test"

    design = source.copy()
    design["role"] = design.crop_id.map(role)
    for cell in CELLS:
        design[cell] = design.crop_id.map(partitions[cell]).fillna("unused")
    design["seed"] = seed
    summary = pd.DataFrame(summaries)
    validate_factorial(design, config)
    for key, value in asdict(config).items():
        summary[key] = value
    return design, summary


def validate_factorial(design: pd.DataFrame, config: FactorialConfig) -> None:
    """Fail closed unless every registered 2C invariant holds."""
    if design.role.isna().any():
        raise RuntimeError("every crop must receive a role")
    test_sets = {cell: frozenset(design.crop_id[design[cell] == "test"]) for cell in CELLS}
    if len(set(test_sets.values())) != 1:
        raise RuntimeError("cells do not share one test set")
    test = design.loc[design.role == "test"]
    if frozenset(test.crop_id) != test_sets[CELLS[0]]:
        raise RuntimeError("test partition differs from the test role")
    test_parents = set(test.parent_id)
    counts = {}
    for cell in CELLS:
        train = design.loc[design[cell] == "train"]
        if (train.role == "test").any() or (train.role == "unseen_background").any():
            raise RuntimeError(f"{cell}: forbidden crops in training")
        per_class = train.groupby("class_name").size()
        if not (per_class == config.crops_per_class).all():
            raise RuntimeError(f"{cell}: class crop budget violated: {per_class.to_dict()}")
        present = set(train.parent_id) & test_parents
        if cell.startswith("leak") and present != test_parents:
            raise RuntimeError(f"{cell}: {len(test_parents - present)} test parents absent from training")
        if cell.startswith("clean") and present:
            raise RuntimeError(f"{cell}: {len(present)} test parents leak into training")
        counts[cell] = train.groupby("class_name").parent_id.nunique()
    for level in ("high", "low"):
        if not counts[f"leak_{level}"].equals(counts[f"clean_{level}"]):
            raise RuntimeError(f"{level}: leak and clean parent counts differ")
    if not (counts["leak_high"] > counts["leak_low"]).all():
        raise RuntimeError("high diversity must contain more parents than low diversity")
    for leak in ("leak", "clean"):
        high = set(design.parent_id[design[f"{leak}_high"] == "train"])
        low = set(design.parent_id[design[f"{leak}_low"] == "train"])
        if not low <= high:
            raise RuntimeError(f"{leak}: low-diversity parents are not nested in high-diversity parents")


def cell_split(design: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Trainer-facing split for one cell (only train and test rows)."""
    if cell not in CELLS:
        raise ValueError(f"Unknown factorial cell: {cell}")
    out = design.loc[design[cell] != "unused", ["crop_id", "parent_id", "class_name", "role"]].copy()
    out["partition"] = design.loc[design[cell] != "unused", cell].to_numpy()
    out["protocol"] = cell
    out["seed"] = int(design.seed.iloc[0])
    return out.sort_values("crop_id").reset_index(drop=True)


def effective_parent_number(train: pd.DataFrame) -> float:
    """exp(Shannon entropy) of the crop distribution over training parents."""
    shares = train.groupby("parent_id").size().to_numpy(dtype=float)
    shares = shares / shares.sum()
    return float(np.exp(-(shares * np.log(shares)).sum()))
