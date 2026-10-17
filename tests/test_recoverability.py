import itertools

import numpy as np
import pandas as pd
import pytest

from cytofair.recoverability import embedding_diversity, parent_recoverability


def probe_frame(n_classes=3, parents_per_class=8):
    rows = []
    for class_index in range(n_classes):
        for parent in range(parents_per_class):
            for crop, partition in enumerate(("fit", "fit", "eval", "eval")):
                rows.append(
                    {
                        "crop_id": f"c{class_index}p{parent}x{crop}",
                        "parent_id": f"c{class_index}p{parent}",
                        "class_name": f"c{class_index}",
                        "probe_partition": partition,
                    }
                )
    return pd.DataFrame(rows)


def test_parent_signal_is_recovered_and_class_only_features_sit_at_the_floor():
    probe = probe_frame()
    rng = np.random.default_rng(0)
    parents = sorted(probe.parent_id.unique())
    parent_centres = {parent: rng.normal(size=16) * 5 for parent in parents}
    class_centres = {name: rng.normal(size=16) * 5 for name in probe.class_name.unique()}
    parent_features = np.stack([parent_centres[p] + rng.normal(size=16) * 0.1 for p in probe.parent_id])
    class_features = np.stack([class_centres[c] + rng.normal(size=16) * 0.1 for c in probe.class_name])
    strong = parent_recoverability(parent_features, probe.crop_id, probe)
    weak = parent_recoverability(class_features, probe.crop_id, probe)
    assert strong["within_class_accuracy"] > 0.9
    assert strong["within_class_ncc_accuracy"] > 0.9
    assert weak["within_class_floor"] == pytest.approx(1 / 8)
    assert weak["within_class_accuracy"] < 0.5
    assert strong["chance"] == pytest.approx(1 / 24)


def test_missing_features_fail_closed():
    probe = probe_frame()
    with pytest.raises(ValueError, match="no features"):
        parent_recoverability(np.zeros((2, 3)), ["x", "y"], probe)


def test_embedding_diversity_matches_brute_force():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(9, 4))
    parents = ["a", "a", "b", "b", "b", "c", "c", "d", "d"]
    result = embedding_diversity(features, parents)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    brute = np.mean([1 - unit[i] @ unit[j] for i, j in itertools.permutations(range(9), 2)])
    assert result["mean_pairwise_crop_cosine_distance"] == pytest.approx(brute)
    assert result["n_parents"] == 4
