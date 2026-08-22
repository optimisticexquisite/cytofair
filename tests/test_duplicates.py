import pandas as pd

from cytofair.duplicates import attach_p3_groups, duplicate_pairs


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

