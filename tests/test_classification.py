import pandas as pd

from cytofair.classification import classification_split
from cytofair.splits import structural_leakage


def source():
    rows = []
    for parent in range(20):
        for crop in range(5):
            rows.append(
                {
                    "crop_id": f"{parent}-{crop}",
                    "parent_id": f"p{parent}",
                    "class_name": f"c{parent % 2}",
                    "crop_path": f"/{parent}-{crop}.bmp",
                    "p3_group_id": f"p{parent}",
                }
            )
    return pd.DataFrame(rows)


def test_protocol_splits_have_expected_leakage():
    frame = source()
    assert structural_leakage(classification_split(frame, "P1", 17)).n_straddled_parents > 0
    assert structural_leakage(classification_split(frame, "P2", 17)).n_straddled_parents == 0
    assert structural_leakage(classification_split(frame, "P3", 17)).n_straddled_parents == 0


def test_size_matched_protocol_matches_p2_training_count():
    frame = source()
    matched = classification_split(frame, "P1-prime", 17)
    grouped = classification_split(frame, "P2", 17)
    assert (matched.partition == "train").sum() == (grouped.partition == "train").sum()

