import numpy as np
import pandas as pd
import pytest

from cytofair.risk import structural_risk
from cytofair.splits import make_grouped, make_p1, structural_leakage


def frame(sizes, classes=None):
    rows = []
    for parent, size in enumerate(sizes):
        for crop in range(size):
            label = classes[parent] if classes is not None else f"c{parent % 3}"
            rows.append({"crop_id": f"{parent}-{crop}", "parent_id": f"p{parent}", "class_name": label})
    return pd.DataFrame(rows)


def test_singleton_dataset_has_no_structural_risk():
    report = structural_risk(frame([1] * 50))
    assert report["expected_parent_straddle"] == 0
    assert report["expected_test_crops_with_training_sibling"] == 0
    assert report["training_diversity_ratio"] == pytest.approx(1.0)
    assert report["class_parent_dependence"] == pytest.approx(1.0)


def test_closed_forms_match_simulated_crop_splits():
    rng = np.random.default_rng(0)
    data = frame(list(rng.integers(1, 12, 300)))
    report = structural_risk(data)
    observed = [structural_leakage(make_p1(data, seed)) for seed in range(20)]
    assert np.mean([o.straddle_fraction for o in observed]) == pytest.approx(
        report["expected_parent_straddle"], abs=0.03
    )
    assert np.mean([o.leaked_test_crop_fraction for o in observed]) == pytest.approx(
        report["expected_test_crops_with_training_sibling"], abs=0.03
    )
    p2 = make_grouped(data, 3)
    assert p2.loc[p2.partition == "train"].parent_id.nunique() == pytest.approx(
        report["expected_training_parents_parent_split"], rel=0.05
    )


def test_class_parent_dependence_is_zero_when_classes_are_mixed_within_parents():
    rows = []
    for parent in range(40):
        for crop, label in enumerate(("a", "b", "a", "b")):
            rows.append({"crop_id": f"{parent}-{crop}", "parent_id": f"p{parent}", "class_name": label})
    report = structural_risk(pd.DataFrame(rows))
    assert report["class_parent_dependence"] == pytest.approx(0.0, abs=1e-12)
    assert report["mean_parent_majority_class_purity"] == pytest.approx(0.5)
