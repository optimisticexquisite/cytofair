import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from cytofair.carriers import (
    ABLATIONS,
    FILL,
    ablation_plan,
    background_grey,
    build_ablation_cache,
    grey,
    illumination_flat,
    lab_statistics,
    low_pass,
    reinhard,
    stain_transfer,
)
from cytofair.image_cache import CropCache, write_array_cache


def _cell(seed: int = 0, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), (225, 215, 200), dtype=np.uint8)
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 4, (90, 60, 140), -1)
    return image


def test_reinhard_identity_and_full_transfer():
    image, donor = _cell(0), _cell(1)
    donor = np.clip(donor.astype(int) + (20, -10, 5), 0, 255).astype(np.uint8)
    assert np.array_equal(reinhard(image, (0, 0, 0), (1, 1, 1), 0.0), image)
    moved = stain_transfer(image, donor)
    target_mean, target_std = lab_statistics(donor)
    mean, std = lab_statistics(moved)
    assert np.allclose(mean, target_mean, atol=1.5)
    assert np.allclose(std, target_std, rtol=0.1, atol=0.5)
    with pytest.raises(ValueError):
        reinhard(image, target_mean, target_std, 1.5)


def test_background_grey_keeps_cell_and_fills_background():
    image = _cell(2)
    out = background_grey(image)
    assert tuple(out[2, 2]) == FILL
    assert np.array_equal(out[48, 48], image[48, 48])


def test_simple_transforms_preserve_shape_and_type():
    image = _cell(3, size=130)
    for transform in (grey, low_pass, illumination_flat):
        out = transform(image)
        assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(grey(image)[..., 0], grey(image)[..., 2])


def test_ablation_plan_and_cache(tmp_path):
    design = pd.DataFrame({
        "crop_id": ["a", "b", "u1", "u2"],
        "parent_id": ["A", "B", "U1", "U2"],
        "class_name": ["x", "y", "x", "y"],
        "role": ["test", "test", "unseen_background", "unseen_background"],
    })
    transplants = pd.DataFrame({
        "recipient_id": ["a", "b"], "condition": ["unseen_same_class"] * 2, "donor_id": ["u1", "u2"],
    })
    plan = ablation_plan(design, transplants)
    assert len(plan) == 2 * len(ABLATIONS)
    stain = plan.loc[plan.ablation == "stain_transfer"].set_index("recipient_id").donor_id
    assert stain.to_dict() == {"a": "u1", "b": "u2"}
    write_array_cache(((({"crop_id": c}), _cell(i)) for i, c in enumerate(design.crop_id)), tmp_path, "base",
                      source={"kind": "test"})
    cache = CropCache(tmp_path, "base", verify=True)
    build_ablation_cache(plan, cache, tmp_path, "ablations")
    ablated = CropCache(tmp_path, "ablations", verify=True)
    assert sorted(ablated.index.crop_id) == sorted(plan.ablated_id)
    assert np.array_equal(ablated.array("a|original"), cache.array("a"))
