import hashlib
from pathlib import Path

import pandas as pd

from cytofair.backbones import ARCHITECTURES, WEIGHT_NAMES, dann_schedule
from cytofair.study2_design import (
    FACTORIAL_CONFIG,
    MECHANISM_SEEDS,
    STUDY2_SEEDS,
    export_protocol_splits,
)


def source(parents=40):
    rows = []
    for parent in range(parents):
        for crop in range(1 + parent % 5):
            rows.append(
                {
                    "crop_id": f"{parent:03d}-{crop}",
                    "parent_id": f"p{parent:03d}",
                    "class_name": f"c{parent % 2}",
                    "p3_group_id": f"g{parent // 2:03d}",
                }
            )
    return pd.DataFrame(rows)


def test_study2_seeds_are_new_and_registered():
    study1 = {17, 29, 43, 71, 101}
    assert not study1 & set(STUDY2_SEEDS)
    assert len(set(STUDY2_SEEDS)) == len(STUDY2_SEEDS)
    assert MECHANISM_SEEDS == STUDY2_SEEDS[:10]
    assert FACTORIAL_CONFIG.crops_per_class == 220


def test_protocol_export_hashes_files_and_flags_identical_p1_prime(tmp_path):
    manifest = export_protocol_splits(source(), tmp_path, dataset="Toy", seeds=(137, 173))
    assert set(manifest.protocol) == {"P1", "P1-prime", "P2", "P3"}
    for row in manifest.itertuples(index=False):
        payload = Path(row.path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row.sha256
    p2 = manifest.loc[manifest.protocol == "P2"]
    assert (p2.straddle_fraction == 0).all()
    prime = manifest.loc[manifest.protocol == "P1-prime"].set_index("seed")
    p1 = manifest.loc[manifest.protocol == "P1"].set_index("seed")
    expected = prime.semantic_sha256 == p1.semantic_sha256
    assert (prime.identical_to_p1 == expected).all()


def test_backbone_registry_and_dann_schedule():
    assert len(ARCHITECTURES) == 12
    assert set(WEIGHT_NAMES) == set(ARCHITECTURES)
    assert dann_schedule(0.0, 1.0) == 0.0
    assert 0.99 < dann_schedule(1.0, 1.0) <= 1.0
    assert dann_schedule(0.5, 0.3) == 0.3 * dann_schedule(0.5, 1.0)
