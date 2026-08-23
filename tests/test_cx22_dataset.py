from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from cytofair.cx22_dataset import materialize_dataset, read_instance_masks, reconstruct_roi


def _write_nested_masks(path: Path, key: str, masks: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        refs = handle.create_group("#refs#")
        references = []
        for index, mask in enumerate(masks):
            references.append(refs.create_dataset(f"mask{index}", data=mask.T).ref)
        inner = refs.create_dataset("inner", (1, len(references)), dtype=h5py.ref_dtype)
        inner[0] = references
        outer = handle.create_dataset(key, (1, 1), dtype=h5py.ref_dtype)
        outer[0, 0] = inner.ref


def test_read_instance_masks_restores_matlab_orientation(tmp_path: Path):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[10:20, 30:50] = 1
    path = tmp_path / "masks.mat"
    _write_nested_masks(path, "cyto_clumps", [mask])
    decoded = read_instance_masks(path, "cyto_clumps")
    assert len(decoded) == 1
    assert decoded[0].shape == (1, 512, 512)
    assert np.array_equal(decoded[0][0], mask)


def test_reconstruct_roi_pastes_at_top_left(tmp_path: Path):
    parent = tmp_path / "parent.png"
    source = np.zeros((8, 10, 3), dtype=np.uint8)
    source[2:6, 3:8] = (12, 34, 56)
    Image.fromarray(source).save(parent)
    image = reconstruct_roi(
        pd.Series(
            {
                "parent_path": str(parent),
                "x": 3,
                "y": 2,
                "width": 5,
                "height": 4,
                "requested_width": 5,
                "requested_height": 4,
            }
        )
    )
    values = np.asarray(image)
    assert image.size == (512, 512)
    assert np.all(values[:4, :5] == (12, 34, 56))
    assert np.all(values[4:, 5:] == 255)


def test_materialize_dataset_writes_complete_manifest(tmp_path: Path):
    labels = tmp_path / "labels" / "Cx22-Pair"
    (labels / "generator").mkdir(parents=True)
    (labels / "generator" / "ImageDataNames.mat").touch()
    cyto = np.zeros((512, 512), dtype=np.uint8)
    nuc = np.zeros((512, 512), dtype=np.uint8)
    cyto[20:80, 30:90], nuc[40:50, 50:60] = 1, 1
    _write_nested_masks(labels / "cyto" / "cyto_clumps.mat", "cyto_clumps", [cyto])
    _write_nested_masks(labels / "nuc" / "nuc_clumps.mat", "nuc_clumps", [nuc])
    parent = tmp_path / "00001.png"
    Image.new("RGB", (100, 80), "pink").save(parent)
    provenance = pd.DataFrame(
        {
            "crop_id": ["Cx22-Pair/0000"],
            "parent_id": ["00001"],
            "class_name": ["segmentation"],
            "subset": ["Cx22-Pair"],
            "x": [0],
            "y": [0],
            "width": [100],
            "height": [80],
            "parent_path": [str(parent)],
        }
    )
    manifest = materialize_dataset(provenance, tmp_path / "labels", tmp_path / "assets", expected_total=1)
    assert len(manifest) == 1
    assert manifest.n_cyto_instances.tolist() == [1]
    assert Path(manifest.iloc[0].crop_path).is_file()
    with np.load(manifest.iloc[0].cyto_instances_path) as archive:
        assert archive["masks"].shape == (1, 512, 512)
