import numpy as np
import pandas as pd
from PIL import Image

from cytofair.image_cache import CropCache, write_array_cache
from cytofair.probe import _ImageMode
from cytofair.transplant import (
    build_transplant_cache,
    cell_removed_background,
    composite,
    foreground_mask,
    transplant_plan,
)


def cell_image(seed: int, size=(40, 48)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((*size, 3), 230, dtype=np.uint8) + rng.integers(0, 10, (*size, 3), dtype=np.uint8)
    image[12:28, 14:34] = (90, 40, 120)  # a dark, saturated "cell"
    return image


def test_mask_and_inpainting_match_the_study1_background_probe():
    image = cell_image(1)
    probe = np.asarray(_ImageMode("background_only")(Image.fromarray(image)))
    np.testing.assert_array_equal(cell_removed_background(image), probe)
    mask = foreground_mask(image)
    assert mask[20, 24] == 1 and mask[2, 2] == 0


def test_composite_keeps_foreground_and_swaps_background():
    recipient = cell_image(2)
    donor = np.zeros((30, 30, 3), dtype=np.uint8)
    mask = foreground_mask(recipient)
    mixed = composite(recipient, donor, mask)
    assert mixed.shape == recipient.shape
    assert np.abs(mixed[20, 24].astype(int) - recipient[20, 24].astype(int)).max() <= 2
    assert mixed[1, 1].max() <= 5  # far background comes from the black donor


def design():
    rows = []
    roles = {
        "a1": ["test", "test", "donor_sibling", "donor_sibling"],
        "b1": ["test", "donor_sibling"],
        "a9": ["unseen_background"] * 3,
        "b9": ["unseen_background"] * 3,
        "a5": ["clean_pool"],
    }
    for parent, parent_roles in roles.items():
        for index, role in enumerate(parent_roles):
            rows.append(
                {"crop_id": f"{parent}-{index}", "parent_id": parent, "class_name": parent[0], "role": role, "seed": 7}
            )
    return pd.DataFrame(rows)


def test_plan_draws_donors_from_the_registered_roles():
    frame = design()
    plan = transplant_plan(frame)
    role = frame.set_index("crop_id").role
    assert plan.groupby("recipient_id").size().eq(5).all()  # 4 conditions + one other class
    sibling = plan.loc[plan.condition == "sibling"]
    assert (sibling.donor_parent == sibling.recipient_parent).all()
    assert sibling.donor_id.map(role).eq("donor_sibling").all()
    unseen = plan.loc[plan.condition.str.startswith("unseen")]
    assert unseen.donor_id.map(role).eq("unseen_background").all()
    same = plan.loc[plan.condition == "unseen_same_class"]
    assert (same.donor_class == same.recipient_class).all()
    other = plan.loc[plan.condition == "unseen_other_class"]
    assert (other.donor_class != other.recipient_class).all()
    pd.testing.assert_frame_equal(plan, transplant_plan(frame))


def test_transplant_cache_renders_every_composite(tmp_path):
    frame = design()
    items = [({"crop_id": crop}, cell_image(index)) for index, crop in enumerate(frame.crop_id)]
    write_array_cache(items, tmp_path, "base", source={"kind": "test"})
    cache = CropCache(tmp_path, "base", verify=True)
    manifest, plan = build_transplant_cache(frame, cache, str(tmp_path), "composites")
    rendered = CropCache(tmp_path, "composites", verify=True)
    assert manifest["n_crops"] == len(plan)
    original = plan.loc[plan.condition == "original"].iloc[0]
    np.testing.assert_array_equal(rendered.array(original.composite_id), cache.array(original.recipient_id))
    assert plan.background_fraction.between(0, 1).all()
