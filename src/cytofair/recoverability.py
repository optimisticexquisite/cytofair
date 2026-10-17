"""Parent-fingerprint recoverability and embedding diversity from frozen features.

Recoverability is measured on the balanced 424-parent probe split (two fitting
and two evaluation crops per parent). Because every SIPaKMeD parent belongs to
one class, a representation that encodes only the class already identifies a
parent with probability ``1 / (parents in that class)``. The primary quantity is
therefore *within-class* identification accuracy, reported next to that
class-only floor; overall 424-way accuracy and nearest-centroid accuracy are
secondary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _aligned(features: np.ndarray, crop_ids, probe: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    position = {crop_id: index for index, crop_id in enumerate(map(str, crop_ids))}
    missing = [crop for crop in probe.crop_id if crop not in position]
    if missing:
        raise ValueError(f"{len(missing)} probe crops have no features")
    order = np.asarray([position[crop] for crop in probe.crop_id])
    return np.asarray(features, dtype=np.float32)[order], probe.reset_index(drop=True)


def parent_recoverability(
    features: np.ndarray, crop_ids, probe: pd.DataFrame, *, C: float = 1.0, logistic: bool = True
) -> dict:
    """Linear-probe parent identification from frozen penultimate features.

    ``logistic=False`` skips the (registered, slower) logistic probe and returns only
    the nearest-centroid measures, used for descriptive Phase 1 summaries.
    """
    required = {"crop_id", "parent_id", "class_name", "probe_partition"}
    missing = sorted(required - set(probe.columns))
    if missing:
        raise ValueError(f"Probe split misses: {missing}")
    values, probe = _aligned(features, crop_ids, probe)
    fit = (probe.probe_partition == "fit").to_numpy()
    evaluate = (probe.probe_partition == "eval").to_numpy()
    parents = sorted(probe.parent_id.unique())
    parent_index = {parent: index for index, parent in enumerate(parents)}
    y = probe.parent_id.map(parent_index).to_numpy()
    parent_class = probe.drop_duplicates("parent_id").set_index("parent_id").class_name
    class_of = np.asarray([parent_class[parent] for parent in parents])

    eval_class = probe.class_name.to_numpy()[evaluate]
    same_class = class_of[None, :] == eval_class[:, None]
    truth = y[evaluate]
    logistic_metrics = {}
    if logistic:
        scaler = StandardScaler().fit(values[fit])
        model = LogisticRegression(C=C, max_iter=5000, tol=1e-4)
        model.fit(scaler.transform(values[fit]), y[fit])
        full = np.full((int(evaluate.sum()), len(parents)), -np.inf)
        full[:, model.classes_] = model.decision_function(scaler.transform(values[evaluate]))
        within = np.where(same_class, full, -np.inf)
        logistic_metrics = {
            "overall_accuracy": float((full.argmax(axis=1) == truth).mean()),
            "within_class_accuracy": float((within.argmax(axis=1) == truth).mean()),
        }

    # Nearest-centroid (cosine) as a hyperparameter-free sensitivity measure.
    normed = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    centroids = np.zeros((len(parents), values.shape[1]), dtype=np.float32)
    for parent, index in parent_index.items():
        centroids[index] = normed[fit & (probe.parent_id.to_numpy() == parent)].mean(axis=0)
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    similarity = normed[evaluate] @ centroids.T
    within_ncc = np.where(same_class, similarity, -np.inf)

    parents_per_class = pd.Series(class_of).value_counts()
    return {
        "n_parents": len(parents),
        "n_fit": int(fit.sum()),
        "n_eval": int(evaluate.sum()),
        **logistic_metrics,
        "within_class_ncc_accuracy": float((within_ncc.argmax(axis=1) == truth).mean()),
        "overall_ncc_accuracy": float((similarity.argmax(axis=1) == truth).mean()),
        "chance": 1.0 / len(parents),
        "within_class_floor": float(np.mean([1.0 / parents_per_class[c] for c in eval_class])),
        "logistic_C": C,
    }


def embedding_diversity(features: np.ndarray, parent_ids) -> dict:
    """Mean pairwise cosine distance among crops and among parent centroids."""
    values = np.asarray(features, dtype=np.float64)
    values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    n = len(values)
    mean_vector = values.mean(axis=0)
    # mean over i != j of (1 - cos) = 1 - (|sum v|^2 - n) / (n (n - 1)) for unit vectors
    crop_distance = 1.0 - (n * n * float(mean_vector @ mean_vector) - n) / (n * (n - 1))
    frame = pd.DataFrame(values)
    frame["parent_id"] = list(parent_ids)
    centroids = frame.groupby("parent_id").mean().to_numpy(copy=True)
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    m = len(centroids)
    centroid_mean = centroids.mean(axis=0)
    parent_distance = 1.0 - (m * m * float(centroid_mean @ centroid_mean) - m) / (m * (m - 1))
    return {
        "n_crops": n,
        "n_parents": m,
        "mean_pairwise_crop_cosine_distance": float(crop_distance),
        "mean_pairwise_parent_centroid_cosine_distance": float(parent_distance),
    }
