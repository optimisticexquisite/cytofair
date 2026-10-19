"""Deterministic export of every Study 2 split before any Study 2 model is trained.

All split files are plain CSVs with an adjacent manifest of SHA-256 hashes,
counts, and structural-leakage measurements. Training jobs refuse a split whose
hash differs from the manifest, so the analysis can only see registered
partitions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .factorial import CELLS, FactorialConfig, build_factorial, cell_split, effective_parent_number
from .probe import balanced_parent_split
from .splits import make_grouped, make_p1, make_p1_prime, structural_leakage
from .transplant import transplant_plan

# Study 1 used 17, 29, 43, 71, 101 and named 137-307 as robustness seeds. None of
# these twenty seeds has been used to train a model before Study 2.
STUDY2_SEEDS = (
    137, 173, 211, 257, 307, 349, 401, 457, 503, 557,
    601, 653, 701, 751, 809, 853, 907, 953, 1009, 1051,
)
MECHANISM_SEEDS = STUDY2_SEEDS[:10]
# Amendment 8 (follow-up). Retraining arms on natural splits use the ten Study 2
# seeds that no mechanism arm used; the carrier-removal factorial uses ten seeds
# never used for any model, each checked feasible for FACTORIAL_CONFIG.
FOLLOWUP_SPLIT_SEEDS = STUDY2_SEEDS[10:]
FOLLOWUP_FACTORIAL_SEEDS = (1103, 1151, 1201, 1249, 1301, 1361, 1409, 1451, 1499, 1553)
PROBE_SPLIT_SEED = 17
# 220 crops per class and a 0.70 low/high parent ratio were feasible for every
# registered seed and for 100/100 unrelated seeds; 240 crops failed for seed 307
# and for 10/100 unrelated seeds, so it was rejected before any training.
FACTORIAL_CONFIG = FactorialConfig(
    evaluation_fraction=0.30,
    donor_parents_per_class=6,
    donor_parent_min_crops=3,
    crops_per_class=220,
    low_to_high_ratio=0.70,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(split: pd.DataFrame) -> str:
    payload = split.sort_values("crop_id")[["crop_id", "partition"]].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.sort_values("crop_id")[list(columns)].to_csv(index=False, lineterminator="\n")
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def _split_record(split: pd.DataFrame, path: Path, sha: str, **fields) -> dict:
    used = split.loc[split.partition.isin(["train", "test"])]
    train = used.loc[used.partition == "train"]
    test = used.loc[used.partition == "test"]
    leakage = structural_leakage(used[["crop_id", "parent_id", "partition"]])
    return {
        **fields,
        "path": path.as_posix(),
        "sha256": sha,
        "semantic_sha256": _semantic_sha256(used),
        "train_crops": len(train),
        "test_crops": len(test),
        "train_parents": train.parent_id.nunique(),
        "test_parents": test.parent_id.nunique(),
        "effective_train_parents": effective_parent_number(train),
        "straddle_fraction": leakage.straddle_fraction,
        "test_crops_with_train_parent": leakage.leaked_test_crop_fraction,
    }


def export_protocol_splits(
    provenance: pd.DataFrame,
    root: str | Path,
    *,
    dataset: str,
    seeds: Iterable[int],
    protocols: Sequence[str] = ("P1", "P1-prime", "P2", "P3"),
) -> pd.DataFrame:
    root = Path(root)
    records = []
    for seed in seeds:
        p1 = make_p1(provenance, seed)
        p2 = make_grouped(provenance, seed, protocol="P2")
        splits = {"P1": p1, "P2": p2, "P1-prime": make_p1_prime(p1, p2, seed)}
        if "P3" in protocols:
            if "p3_group_id" not in provenance.columns:
                raise ValueError("P3 requires p3_group_id")
            splits["P3"] = make_grouped(provenance, seed, protocol="P3", group_column="p3_group_id")
        for protocol in protocols:
            split = splits[protocol]
            safe = protocol.lower().replace("-", "_")
            path = root / f"{dataset.lower()}_{safe}_seed{seed}.csv"
            sha = _write_csv(split, path, ("crop_id", "parent_id", "class_name", "partition", "protocol", "seed"))
            records.append(_split_record(split, path, sha, dataset=dataset, protocol=protocol, seed=seed))
    manifest = pd.DataFrame(records)
    # A P1-prime split equal to P1 needs no separate training run: identical
    # partitions, seed, and recipe reproduce the same deterministic result.
    p1_hash = manifest.loc[manifest.protocol == "P1"].set_index("seed").semantic_sha256
    manifest["identical_to_p1"] = (manifest.protocol == "P1-prime") & (
        manifest.semantic_sha256 == manifest.seed.map(p1_hash)
    )
    return manifest


def bbox_overlap_fraction(boxes: pd.DataFrame, split: pd.DataFrame) -> float:
    """Fraction of test crops sharing at least one source pixel with a training crop."""
    merged = split.merge(boxes[["crop_id", "x", "y", "width", "height"]], on="crop_id", how="left")
    if merged[["x", "y", "width", "height"]].isna().any().any():
        raise ValueError("every split crop needs a recovered box")
    overlapping = set()
    for _, group in merged.groupby("parent_id"):
        train = group.loc[group.partition == "train"]
        test = group.loc[group.partition == "test"]
        if train.empty or test.empty:
            continue
        tx0, ty0 = train.x.to_numpy(), train.y.to_numpy()
        tx1, ty1 = tx0 + train.width.to_numpy(), ty0 + train.height.to_numpy()
        for row in test.itertuples(index=False):
            hit = (np.maximum(tx0, row.x) < np.minimum(tx1, row.x + row.width)) & (
                np.maximum(ty0, row.y) < np.minimum(ty1, row.y + row.height)
            )
            if hit.any():
                overlapping.add(row.crop_id)
    n_test = int((split.partition == "test").sum())
    return len(overlapping) / n_test if n_test else 0.0


def export_factorial(
    provenance: pd.DataFrame,
    root: str | Path,
    *,
    seeds: Iterable[int],
    config: FactorialConfig = FACTORIAL_CONFIG,
    boxes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    root = Path(root)
    records, designs = [], {}
    for seed in seeds:
        design, _ = build_factorial(provenance, seed, config)
        designs[seed] = design
        design_path = root / f"sipakmed_2c_design_seed{seed}.csv"
        design_sha = _write_csv(design, design_path, ("crop_id", "parent_id", "class_name", "role", *CELLS, "seed"))
        for cell in CELLS:
            split = cell_split(design, cell)
            path = root / f"sipakmed_2c_{cell}_seed{seed}.csv"
            sha = _write_csv(split, path, ("crop_id", "parent_id", "class_name", "role", "partition", "protocol", "seed"))
            train = split.loc[split.partition == "train"]
            test = split.loc[split.partition == "test"]
            siblings = test.parent_id.map(train.groupby("parent_id").size()).fillna(0)
            record = _split_record(
                split,
                path,
                sha,
                dataset="SIPaKMeD",
                protocol=cell,
                seed=seed,
                design_path=design_path.as_posix(),
                design_sha256=design_sha,
                mean_train_siblings_per_test_crop=float(siblings.mean()),
            )
            if boxes is not None:
                record["test_crops_with_literal_overlap"] = bbox_overlap_fraction(boxes, split)
            records.append(record)
    return pd.DataFrame(records), designs


def export_probe_split(provenance: pd.DataFrame, path: str | Path, seed: int = PROBE_SPLIT_SEED) -> str:
    split, _ = balanced_parent_split(provenance, seed)
    split = split.rename(columns={"partition": "probe_partition"})
    split["probe_partition"] = split.probe_partition.map({"train": "fit", "test": "eval"})
    classes = provenance.drop_duplicates("parent_id").set_index("parent_id").class_name
    split["class_name"] = split.parent_id.map(classes)
    return _write_csv(
        split, Path(path), ("crop_id", "parent_id", "class_name", "label", "probe_partition", "seed")
    )


def export_transplant_plans(designs: dict[int, pd.DataFrame], root: str | Path) -> pd.DataFrame:
    root = Path(root)
    records = []
    for seed, design in designs.items():
        plan = transplant_plan(design)
        path = root / f"sipakmed_2b_plan_seed{seed}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = plan.sort_values("composite_id").to_csv(index=False, lineterminator="\n")
        path.write_text(payload)
        records.append(
            {
                "seed": seed,
                "path": path.as_posix(),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "composites": len(plan),
                "recipients": plan.recipient_id.nunique(),
            }
        )
    return pd.DataFrame(records)
