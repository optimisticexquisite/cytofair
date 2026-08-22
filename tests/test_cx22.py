from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from cytofair.cx22 import add_parent_dimensions, recover_metadata, subset_parent_overlap


def _write_string_object(path: Path, names: list[str]):
    lengths = np.asarray([len(name) for name in names], dtype="<u8")
    encoded = "".join(names).encode("utf-16le")
    encoded += bytes((-len(encoded)) % 8)
    packed = np.frombuffer(encoded, dtype="<u8")
    payload = np.concatenate(
        [np.asarray([1, 2, len(names), 1], dtype="<u8"), lengths, packed]
    )
    with h5py.File(path, "w") as handle:
        refs = handle.create_group("#refs#")
        refs.create_dataset("c", data=payload.reshape(-1, 1))
        public = handle.create_dataset(
            "ImageDataNames", data=np.asarray([[3707764736, 2, 1, 1, 1, 1]], dtype=np.uint32)
        )
        public.attrs["MATLAB_class"] = np.bytes_(b"string")


def _write_numeric(path: Path, key: str, values: np.ndarray):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=values.T)


def test_cx22_mcos_strings_and_rois(tmp_path: Path):
    generator = tmp_path / "Cx22-Pair" / "generator"
    generator.mkdir(parents=True)
    _write_string_object(generator / "ImageDataNames.mat", ["00001", "00001", "00002"])
    _write_numeric(
        generator / "ROIs_W_H.mat",
        "ROIs_W_H",
        np.asarray([[100, 200], [110, 210], [120, 220]], dtype=float),
    )
    _write_numeric(
        generator / "ROIs_x_y.mat",
        "ROIs_x_y",
        np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float),
    )
    frame = recover_metadata(tmp_path / "Cx22-Pair")
    assert len(frame) == 3
    assert frame.parent_id.tolist() == ["00001", "00001", "00002"]
    assert frame[["x", "y", "width", "height"]].iloc[1].tolist() == [3, 4, 110, 210]


def test_subset_overlap_counts_shared_parents():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "subset": ["a", "a", "b", "b"],
            "parent_id": ["p1", "p2", "p2", "p3"],
        }
    )
    overlap = subset_parent_overlap(frame).iloc[0]
    assert overlap.shared_parents == 1
    assert overlap.jaccard == 1 / 3


def test_parent_dimensions_clip_like_matlab_generator(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 80)).save(image_root / "00001.png")
    import pandas as pd

    frame = pd.DataFrame(
        {
            "source_name": ["00001.png"],
            "x": [90],
            "y": [70],
            "width": [20],
            "height": [15],
        }
    )
    out = add_parent_dimensions(frame, image_root)
    assert out[["width", "height", "parent_width", "parent_height"]].iloc[0].tolist() == [
        10,
        10,
        100,
        80,
    ]
