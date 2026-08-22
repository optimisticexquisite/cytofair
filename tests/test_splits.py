import numpy as np
import pandas as pd

from cytofair.splits import (
    analytic_straddle,
    export_protocol_splits,
    make_grouped,
    make_p1,
    make_p1_prime,
    monte_carlo_straddle_interval,
    structural_leakage,
)


def source(parents=20, crops_per_parent=5):
    rows = []
    for parent in range(parents):
        for crop in range(crops_per_parent):
            rows.append(
                {
                    "crop_id": f"{parent}-{crop}",
                    "parent_id": f"p{parent}",
                    "class_name": f"c{parent % 2}",
                }
            )
    return pd.DataFrame(rows)


def test_analytic_formula_and_monte_carlo_agree():
    sizes = [4] * 100
    expected = 1 - 0.8**4 - 0.2**4
    assert analytic_straddle(sizes) == expected
    low, high = monte_carlo_straddle_interval(sizes, repetitions=5000)
    assert low < expected < high


def test_grouped_split_has_no_parent_leakage():
    split = make_grouped(source(), 17)
    assert structural_leakage(split).n_straddled_parents == 0


def test_random_split_leaks_and_size_match_is_exact():
    frame = source()
    p1 = make_p1(frame, 17)
    p2 = make_grouped(frame, 17)
    p1_prime = make_p1_prime(p1, p2, 17)
    assert structural_leakage(p1).n_straddled_parents > 0
    assert (p1_prime.partition == "train").sum() == (p2.partition == "train").sum()
    assert set(p1_prime.loc[p1_prime.partition == "test", "crop_id"]) == set(
        p1.loc[p1.partition == "test", "crop_id"]
    )


def test_grouped_fold_never_makes_p1_prime_upsample():
    frame = source(parents=21, crops_per_parent=5)
    p1 = make_p1(frame, 101)
    p2 = make_grouped(frame, 101)
    assert (p2.partition == "train").sum() <= (p1.partition == "train").sum()
    assert (make_p1_prime(p1, p2, 101).partition == "train").sum() == (
        p2.partition == "train"
    ).sum()


def test_herlev_style_unique_parent_is_negative_control():
    frame = source(parents=20, crops_per_parent=1)
    split = make_p1(frame, 17)
    metrics = structural_leakage(split)
    assert metrics.straddle_fraction == 0
    assert metrics.leaked_test_crop_fraction == 0
    assert np.isclose(analytic_straddle([1] * 20), 0)


def test_exported_split_assets_cover_all_protocols(tmp_path):
    frame = source()
    frame["p3_group_id"] = frame.parent_id
    manifest = export_protocol_splits(frame, tmp_path, dataset="toy", seeds=[17])
    assert set(manifest.protocol) == {"P1", "P1-prime", "P2", "P3"}
    assert manifest.sha256.str.len().eq(64).all()
    assert (tmp_path / "toy_manifest.csv").is_file()
