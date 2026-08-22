import numpy as np
import pandas as pd
from PIL import Image

from cytofair.probe import _ImageMode, _provenance_sha256, balanced_parent_split


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


def test_all_preregistered_probe_modes_return_rgb_images():
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    for mode in ("rgb", "grayscale", "stain_normalized", "background_only", "highpass"):
        transformed = _ImageMode(mode)(image)
        assert transformed.mode == "RGB"
        assert transformed.size == image.size


def test_provenance_hash_is_order_invariant():
    frame = pd.DataFrame(
        {"crop_id": ["b", "a"], "parent_id": ["p", "q"], "class_name": ["x", "y"]}
    )
    assert _provenance_sha256(frame) == _provenance_sha256(frame.iloc[::-1])
