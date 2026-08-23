import numpy as np
import pandas as pd
import pytest

from cytofair.instance_segmentation import matched_instance_metrics
from cytofair.segmentation import binary_channel_metrics, segmentation_split


def test_official_cx22_split_is_parent_disjoint_and_excludes_pair():
    train = pd.DataFrame(
        {
            "crop_id": [f"train/{i}" for i in range(400)],
            "parent_id": [f"tr{i}" for i in range(400)],
            "subset": "Cx22-Multi-Train",
        }
    )
    test = pd.DataFrame(
        {
            "crop_id": [f"test/{i}" for i in range(100)],
            "parent_id": [f"te{i}" for i in range(100)],
            "subset": "Cx22-Multi-Test",
        }
    )
    pair = pd.DataFrame(
        {"crop_id": ["pair/0"], "parent_id": ["tr0"], "subset": ["Cx22-Pair"]}
    )
    frame = pd.concat([train, test, pair], ignore_index=True).assign(class_name="segmentation")
    split = segmentation_split(frame, "OFFICIAL", 17)
    assert split.partition.value_counts().to_dict() == {"train": 400, "test": 100}
    assert "pair/0" not in set(split.crop_id)


def test_binary_channel_metrics_are_exact():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[[[1, 0], [0, 0]], [[1, 1], [0, 0]]]], dtype=torch.float32)
    prediction = torch.tensor([[[[0.9, 0.1], [0.1, 0.1]], [[0.9, 0.1], [0.1, 0.1]]]])
    dice, iou = binary_channel_metrics(prediction, target)
    assert dice[0, 0] == 1
    assert np.isclose(float(dice[0, 1]), 2 / 3)
    assert np.isclose(float(iou[0, 1]), 1 / 2)


def test_matched_instance_metrics_assign_zero_to_unmatched_target():
    torch = pytest.importorskip("torch")
    predictions = torch.zeros((1, 4, 4))
    predictions[0, :2, :2] = 1
    targets = torch.zeros((2, 4, 4))
    targets[0, :2, :2] = 1
    targets[1, 2:, 2:] = 1
    dice, iou = matched_instance_metrics(predictions, torch.tensor([0.9]), targets)
    assert sorted(dice.tolist()) == [0.0, 1.0]
    assert sorted(iou.tolist()) == [0.0, 1.0]
