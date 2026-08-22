import pandas as pd

from cytofair.probe import balanced_parent_split


def test_probe_split_is_balanced_and_has_exact_chance():
    rows = []
    for parent, count in (("a", 4), ("b", 5), ("c", 7)):
        for crop in range(count):
            rows.append(
                {
                    "crop_id": f"{parent}-{crop}",
                    "parent_id": parent,
                    "crop_path": f"/{parent}-{crop}.bmp",
                }
            )
    split, chance = balanced_parent_split(pd.DataFrame(rows), 17)
    assert chance == 1 / 3
    assert len(split) == 12
    counts = split.groupby(["partition", "label"]).size()
    assert (counts == 2).all()

