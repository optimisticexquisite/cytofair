import pandas as pd
import pytest

from cytofair.factorial import (
    CELLS,
    FactorialConfig,
    build_factorial,
    cell_split,
    effective_parent_number,
)


def provenance(classes=("a", "b"), parents_per_class=60, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for class_name in classes:
        for parent in range(parents_per_class):
            size = int(rng.integers(1, 9))
            for crop in range(size):
                rows.append(
                    {
                        "crop_id": f"{class_name}{parent:03d}-{crop}",
                        "parent_id": f"{class_name}{parent:03d}",
                        "class_name": class_name,
                    }
                )
    return pd.DataFrame(rows)


CONFIG = FactorialConfig(
    evaluation_fraction=0.30,
    donor_parents_per_class=3,
    donor_parent_min_crops=3,
    crops_per_class=60,
    low_to_high_ratio=0.70,
)


def test_cells_share_test_set_and_match_budget_and_parent_counts():
    design, summary = build_factorial(provenance(), 137, CONFIG)
    tests = {cell: frozenset(design.crop_id[design[cell] == "test"]) for cell in CELLS}
    assert len(set(tests.values())) == 1
    test_parents = set(design.parent_id[design.role == "test"])
    for cell in CELLS:
        train = design.loc[design[cell] == "train"]
        assert (train.groupby("class_name").size() == CONFIG.crops_per_class).all()
        present = set(train.parent_id) & test_parents
        assert present == (test_parents if cell.startswith("leak") else set())
        assert not train.role.isin(["test", "unseen_background"]).any()
    counts = summary.pivot_table(index="class_name", columns="cell", values="train_parents")
    assert (counts.leak_high == counts.clean_high).all()
    assert (counts.leak_low == counts.clean_low).all()
    assert (counts.leak_high > counts.leak_low).all()


def test_design_is_deterministic_and_seed_dependent():
    frame = provenance()
    first, _ = build_factorial(frame, 137, CONFIG)
    again, _ = build_factorial(frame.sample(frac=1, random_state=3), 137, CONFIG)
    other, _ = build_factorial(frame, 173, CONFIG)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), again.reset_index(drop=True))
    assert not first[list(CELLS)].equals(other[list(CELLS)])


def test_low_diversity_parents_are_nested_in_high_diversity_parents():
    design, _ = build_factorial(provenance(), 211, CONFIG)
    for leak in ("leak", "clean"):
        high = set(design.parent_id[design[f"{leak}_high"] == "train"])
        low = set(design.parent_id[design[f"{leak}_low"] == "train"])
        assert low < high


def test_infeasible_budget_fails_closed():
    with pytest.raises(ValueError):
        build_factorial(provenance(), 137, FactorialConfig(crops_per_class=10_000))


def test_multi_class_parent_is_rejected():
    frame = provenance()
    frame.loc[frame.index[0], "class_name"] = "b" if frame.class_name.iloc[0] == "a" else "a"
    with pytest.raises(ValueError, match="one class per parent"):
        build_factorial(frame, 137, CONFIG)


def test_cell_split_and_effective_parent_number():
    design, _ = build_factorial(provenance(), 257, CONFIG)
    split = cell_split(design, "clean_low")
    assert set(split.partition) == {"train", "test"}
    assert split.protocol.unique().tolist() == ["clean_low"]
    uniform = pd.DataFrame({"parent_id": ["a", "a", "b", "b"]})
    assert effective_parent_number(uniform) == pytest.approx(2.0)
