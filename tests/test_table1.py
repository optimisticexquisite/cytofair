import pandas as pd

from cytofair.table1 import structural_table


def test_table1_confirms_p3_p2_p1_order_and_control():
    rows = []
    for parent in range(10):
        for crop in range(3):
            rows.append(
                {
                    "crop_id": f"{parent}-{crop}",
                    "parent_id": str(parent),
                    "p3_group_id": str(parent),
                    "class_name": "a" if parent < 5 else "b",
                }
            )
    control = pd.DataFrame(
        {
            "crop_id": [f"c{i}" for i in range(10)],
            "parent_id": [f"c{i}" for i in range(10)],
            "class_name": ["a"] * 5 + ["b"] * 5,
        }
    )
    summary, seeds = structural_table({"cropped": pd.DataFrame(rows), "control": control})
    assert summary.protocol_ordering_passed.all()
    assert summary.loc[summary.dataset == "control", "negative_control_passed"].item()
    assert (seeds.p3_straddle_fraction == 0).all()
