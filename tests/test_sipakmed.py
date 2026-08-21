from pathlib import Path

import cv2
import numpy as np

from cytofair.provenance import validate_provenance
from cytofair.sipakmed import bbox_iou_pairs, recover_branch_a, recover_branch_b


def test_sipakmed_filename_and_template_branches(tmp_path: Path):
    class_root = tmp_path / "im_Parabasal"
    crop_root = class_root / "CROPPED"
    crop_root.mkdir(parents=True)
    rng = np.random.default_rng(4)
    parent = rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)
    cv2.imwrite(str(class_root / "001.bmp"), parent)
    cv2.imwrite(str(crop_root / "001_01.bmp"), parent[10:40, 20:60])
    cv2.imwrite(str(crop_root / "001_02.bmp"), parent[30:60, 35:75])

    branch_a = recover_branch_a(tmp_path)
    assert branch_a.parent_id.nunique() == 1
    branch_b = recover_branch_b(branch_a)
    assert branch_b.accepted.all()
    assert set(zip(branch_b.x, branch_b.y)) == {(20, 10), (35, 30)}
    validation = validate_provenance(
        branch_b, expected_total=2, branch_a=branch_a, branch_b=branch_b, require_bbox=True
    )
    assert validation.passed
    overlaps = bbox_iou_pairs(branch_b)
    assert len(overlaps) == 1
    assert overlaps.iloc[0].intersection_pixels > 0

