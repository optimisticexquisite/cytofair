#!/usr/bin/env python3
"""Registered secondary (non-confirmatory) Study 2 analyses.

* M1: within-crop P1 - P2 accuracy over crops tested under both protocols.
* M2: within-P1 accuracy versus training siblings and literal pixel overlap.
* Phase 1 ranking stability, P3 <= P2 <= P1 ordering, pooled confusion matrices,
  calibration, and nearest-centroid parent recoverability (descriptive).
* 2C embedding diversity in the retrained fingerprint-encoder space.
* The approximate, model-based decomposition of the Phase 1 gap.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cytofair.recoverability import embedding_diversity
from cytofair.statistics import ranking_stability, verify_protocol_order

NATURAL_P1_PARENTS, NATURAL_P2_PARENTS = 914, 773
FACTORIAL_HIGH_PARENTS, FACTORIAL_LOW_PARENTS = 720, 504
NATURAL_P1_LEAK_RATE = 0.917


def crop_level(crops: pd.DataFrame, repetitions: int = 10_000) -> dict:
    by_crop = crops.groupby(["crop_id", "protocol"]).correct.agg(["mean", "sum", "count"]).unstack("protocol")
    both = by_crop.dropna(subset=[("mean", "P1"), ("mean", "P2")])
    difference = (both[("mean", "P1")] - both[("mean", "P2")]).to_numpy()
    rng = np.random.default_rng(20261018)
    boot = np.array([rng.choice(difference, len(difference)).mean() for _ in range(repetitions)])
    m1 = {
        "crops": len(difference),
        "mean_within_crop_difference_pp": 100 * float(difference.mean()),
        "ci95_pp": [100 * float(v) for v in np.quantile(boot, [0.025, 0.975])],
    }
    p2_rate = (both[("sum", "P2")] + 0.5) / (both[("count", "P2")] + 1.0)
    p1 = crops.loc[(crops.protocol == "P1") & crops.crop_id.isin(both.index)].copy()
    p1["difficulty"] = p1.crop_id.map(np.log(p2_rate / (1 - p2_rate)))
    p1["log_siblings"] = np.log1p(p1.train_siblings.astype(float))
    p1["overlap"] = p1.literal_overlap.astype(float)
    design = pd.get_dummies(p1[["overlap", "log_siblings", "difficulty"]].assign(label=p1.label.astype(str)),
                            columns=["label"], drop_first=True, dtype=float)
    model = sm.GLM(p1.correct.astype(float), sm.add_constant(design), family=sm.families.Binomial())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p1.crop_id)[0]})
    m2 = {
        term: {"log_odds": float(fit.params[term]), "se": float(fit.bse[term]), "p": float(fit.pvalues[term])}
        for term in ("overlap", "log_siblings", "difficulty")
    }
    m2["observations"] = int(fit.nobs)
    return {"M1": m1, "M2": m2}


def pooled_confusion(runs: pd.DataFrame) -> dict:
    pooled = {}
    for protocol, group in runs.groupby("protocol"):
        matrices = [np.asarray(json.loads(Path(path).read_text())["confusion_matrix"]) for path in group.run_json]
        pooled[protocol] = np.sum(matrices, axis=0).tolist()
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--mechanism", required=True, help="prepare_mechanism.py output directory")
    parser.add_argument("--fingerprint", required=True, help="fingerprint_embeddings.npz")
    parser.add_argument("--h1", required=True, help="protocol_lmm.R JSON for H1")
    parser.add_argument("--mechanism-json", required=True, help="mechanism_lmm.R JSON")
    parser.add_argument("--splits", default="splits/study2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = {}

    phase1 = pd.read_csv(Path(args.runs) / "phase1_runs.csv")
    out["ranking_P1_vs_P2"] = ranking_stability(phase1.assign(dataset="SIPaKMeD"), value="accuracy").to_dict("records")
    out["ordering_P3_le_P2_le_P1"] = verify_protocol_order(
        phase1, value="accuracy", by=["architecture"]
    ).to_dict("records")
    out["calibration_by_protocol"] = phase1.groupby("protocol")[["nll", "ece_15"]].mean().to_dict("index")
    out["pooled_confusion_matrices"] = pooled_confusion(phase1)

    crops = pd.read_parquet(Path(args.mechanism) / "phase1_crop_predictions.parquet")
    out.update(crop_level(crops))

    archive = np.load(args.fingerprint)
    embeddings = pd.DataFrame(archive["features"].astype(np.float32), index=archive["crop_id"].astype(str))
    diversity = []
    manifest = pd.read_csv(Path(args.splits) / "study2_split_manifest.csv")
    selected = manifest.loc[
        (manifest.dataset == "SIPaKMeD")
        & (manifest.protocol.isin(["P1", "P2", "leak_high", "leak_low", "clean_high", "clean_low"]))
        & manifest.seed.isin(manifest.loc[manifest.protocol == "leak_high", "seed"])
    ]
    for row in selected.itertuples(index=False):
        split = pd.read_csv(row.path)
        train = split.loc[split.partition == "train"]
        diversity.append({"protocol": row.protocol, "seed": row.seed,
                          **embedding_diversity(embeddings.loc[train.crop_id].to_numpy(), train.parent_id)})
    out["embedding_diversity"] = pd.DataFrame(diversity).groupby("protocol").mean(numeric_only=True).to_dict("index")

    h1 = json.loads(Path(args.h1).read_text())
    mechanism = json.loads(Path(args.mechanism_json).read_text())
    leak = mechanism["H3"]["estimate"]
    diversity_effect = mechanism["diversity_effect_clean_secondary"]["estimate"]
    scale = math.log(NATURAL_P1_PARENTS / NATURAL_P2_PARENTS) / math.log(FACTORIAL_HIGH_PARENTS / FACTORIAL_LOW_PARENTS)
    out["approximate_gap_decomposition"] = {
        "phase1_gap_pp": h1["primary_estimate"],
        "leakage_part_pp": leak * NATURAL_P1_LEAK_RATE,
        "diversity_part_pp": diversity_effect * scale,
        "diversity_log_scale_factor": scale,
        "note": "model-based approximation; 2C trains on 1,100 crops versus 3,239 in Phase 1",
    }
    Path(args.output).write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("M1", "approximate_gap_decomposition")}, indent=2, default=float))


if __name__ == "__main__":
    main()
