import numpy as np
import pytest

torch = pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")

from cytofair.gpu_augment import GpuCrops, random_resized_crop_params


@pytest.mark.parametrize("shape", [(191, 215), (48, 300), (553, 62), (100, 100)])
def test_crop_parameters_match_torchvision_exactly(shape):
    height, width = shape
    image = torch.zeros(3, height, width)
    for seed in range(25):
        torch.manual_seed(seed)
        expected = transforms.RandomResizedCrop.get_params(image, [0.8, 1.0], [3 / 4, 4 / 3])
        generator = torch.Generator().manual_seed(seed)
        assert random_resized_crop_params(height, width, generator) == tuple(expected)


def test_batches_are_normalised_and_deterministic():
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 255, (40 + i, 50, 3), dtype=np.uint8) for i in range(4)]
    store = GpuCrops(arrays, torch.device("cpu"))
    first = store.train_batch([0, 1, 2, 3], torch.Generator().manual_seed(5))
    second = store.train_batch([0, 1, 2, 3], torch.Generator().manual_seed(5))
    assert first.shape == (4, 3, 224, 224)
    assert torch.equal(first, second)
    evaluated = store.eval_batch([0, 1])
    assert evaluated.shape == (2, 3, 224, 224)
    assert torch.isfinite(evaluated).all()
