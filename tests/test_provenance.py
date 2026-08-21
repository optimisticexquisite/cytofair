import pandas as pd

from cytofair.provenance import branch_agreement, validate_provenance


def example_map():
    return pd.DataFrame(
        {
            "crop_id": ["a", "b"],
            "parent_id": ["p", "p"],
            "class_name": ["x", "x"],
            "method": ["test", "test"],
            "x": [0, 5],
            "y": [0, 5],
            "width": [10, 5],
            "height": [10, 5],
            "parent_width": [20, 20],
            "parent_height": [20, 20],
        }
    )


def test_provenance_accepts_complete_contained_map():
    frame = example_map()
    validation = validate_provenance(
        frame,
        expected_total=2,
        expected_parent_counts={"x": 1},
        branch_a=frame,
        branch_b=frame,
        require_bbox=True,
    )
    assert validation.passed


def test_provenance_rejects_bbox_past_parent():
    frame = example_map()
    frame.loc[1, "x"] = 18
    validation = validate_provenance(frame, require_bbox=True)
    assert not validation.passed
    assert not validation.frame().set_index("name").loc["bbox_within_parent", "passed"]


def test_branch_agreement_uses_crop_intersection():
    left = example_map()
    right = example_map()
    right.loc[1, "parent_id"] = "other"
    agreement, count = branch_agreement(left, right)
    assert count == 2
    assert agreement == 0.5

