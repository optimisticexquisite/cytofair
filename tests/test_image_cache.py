import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cytofair.image_cache import CropCache, build_file_cache, build_window_cache, window_origin


def test_file_cache_round_trips_exact_pixels(tmp_path):
    rng = np.random.default_rng(0)
    rows, arrays = [], {}
    for index, size in enumerate(((10, 12), (7, 5), (3, 9))):
        pixels = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        path = tmp_path / f"crop{index}.png"
        Image.fromarray(pixels).save(path)
        arrays[f"c{index}"] = pixels
        rows.append({"crop_id": f"c{index}", "crop_path": str(path), "parent_id": "p", "class_name": "k"})
    manifest = build_file_cache(pd.DataFrame(rows), tmp_path / "cache", "test")
    cache = CropCache(tmp_path / "cache", "test", verify=True)
    assert manifest["n_crops"] == 3
    for crop_id, pixels in arrays.items():
        np.testing.assert_array_equal(cache.array(crop_id), pixels)
        assert cache.image(crop_id).mode == "RGB"


def test_cache_verification_detects_tampering(tmp_path):
    rows = []
    path = tmp_path / "one.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
    rows.append({"crop_id": "one", "crop_path": str(path), "parent_id": "p", "class_name": "k"})
    build_file_cache(pd.DataFrame(rows), tmp_path, "t")
    manifest_path = tmp_path / "t_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pixels_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verification"):
        CropCache(tmp_path, "t", verify=True)


def test_window_origin_shifts_inside_the_image():
    assert window_origin(50, 100, 1376) == 0
    assert window_origin(10, 100, 1376) == 0
    assert window_origin(1370, 100, 1376) == 1276
    assert window_origin(600, 100, 1376) == 550
    with pytest.raises(ValueError):
        window_origin(5, 100, 50)


def test_window_cache_cuts_centred_windows(tmp_path):
    source = np.arange(30 * 40 * 3, dtype=np.uint32).reshape(30, 40, 3).astype(np.uint8)
    Image.fromarray(source).save(tmp_path / "img.png")
    cells = pd.DataFrame(
        {
            "crop_id": ["a", "b"],
            "source_name": ["img.png", "img.png"],
            "center_x": [20, 1],
            "center_y": [15, 29],
            "parent_id": ["p", "p"],
            "class_name": ["k", "k"],
        }
    )
    build_window_cache(cells, tmp_path, tmp_path / "cache", "w", size=10)
    cache = CropCache(tmp_path / "cache", "w")
    np.testing.assert_array_equal(cache.array("a"), source[10:20, 15:25])
    np.testing.assert_array_equal(cache.array("b"), source[20:30, 0:10])
    assert set(cache.index[["x", "y"]].itertuples(index=False, name=None)) == {(15, 10), (0, 20)}
