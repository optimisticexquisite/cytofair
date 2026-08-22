import cv2
import numpy as np
import pandas as pd

from cytofair.duplicates import attach_p3_groups, duplicate_pairs, synthetic_detector_check


def test_consensus_hash_pair_and_parent_union():
    hashes = pd.DataFrame(
        {
            "crop_id": ["a", "b", "c"],
            "parent_id": ["p1", "p2", "p3"],
            "phash": [0, 1, 2**63 - 1],
            "dhash": [0, 1, 2**63 - 1],
            "pixel_sha256": ["x", "y", "z"],
        }
    )
    pairs = duplicate_pairs(hashes, phash_threshold=1, dhash_threshold=1)
    assert len(pairs) == 1
    provenance = pd.DataFrame(
        {
            "crop_id": ["a", "b", "c"],
            "parent_id": ["p1", "p2", "p3"],
            "class_name": ["x", "x", "x"],
            "method": ["m", "m", "m"],
        }
    )
    grouped = attach_p3_groups(provenance, pairs)
    assert grouped.set_index("parent_id").p3_group_id["p1"] == grouped.set_index(
        "parent_id"
    ).p3_group_id["p2"]
    assert grouped.set_index("parent_id").p3_group_id["p3"] != grouped.set_index(
        "parent_id"
    ).p3_group_id["p1"]


def test_synthetic_detector_reports_precision_and_recall(tmp_path):
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(base, (64, 64), 30, (220, 120, 80), -1)
    negative = np.zeros_like(base)
    cv2.line(negative, (0, 0), (127, 127), (255, 255, 255), 8)
    base_path, negative_path = tmp_path / "base.png", tmp_path / "negative.png"
    cv2.imwrite(str(base_path), base)
    cv2.imwrite(str(negative_path), negative)
    result = synthetic_detector_check(
        base_path, tmp_path / "injected", negative_paths=[negative_path]
    )
    assert result["positive_examples"] == 3
    assert result["negative_examples"] == 1
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
