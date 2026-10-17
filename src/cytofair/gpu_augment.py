"""GPU-resident crops and the Study 1 augmentation recipe executed on the GPU.

The training host caps each user at four CPU cores, so per-image PIL transforms
in DataLoader workers would starve four GPUs. Instead every crop of a split is
uploaded once as a ``uint8`` CHW tensor and the registered augmentation is
applied on the device:

``RandomResizedCrop(224, scale, ratio=(3/4, 4/3))`` -> horizontal flip (p=0.5)
-> vertical flip (p=0.5) -> ImageNet normalisation.

Crop parameters are drawn with an exact port of torchvision's
``RandomResizedCrop.get_params`` from an explicit CPU generator, so the sampled
geometry is identical to torchvision's for the same random stream. Resizing uses
antialiased bilinear interpolation, torch's implementation of PIL's resampling
filter; values can differ from PIL only by uint8 rounding.
"""

from __future__ import annotations

import math

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def random_resized_crop_params(
    height: int,
    width: int,
    generator,
    scale: tuple[float, float] = (0.8, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> tuple[int, int, int, int]:
    """Port of ``torchvision.transforms.RandomResizedCrop.get_params``."""
    import torch

    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1], generator=generator).item()
        # float32 exp, exactly as torchvision computes it, so w/h round identically.
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(float(log_ratio[0]), float(log_ratio[1]), generator=generator)
        ).item()
        w = round(math.sqrt(target_area * aspect_ratio))
        h = round(math.sqrt(target_area / aspect_ratio))
        if 0 < w <= width and 0 < h <= height:
            i = int(torch.randint(0, height - h + 1, size=(1,), generator=generator).item())
            j = int(torch.randint(0, width - w + 1, size=(1,), generator=generator).item())
            return i, j, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = round(w / min(ratio))
    elif in_ratio > max(ratio):
        h = height
        w = round(h * max(ratio))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


class GpuCrops:
    """Variable-size uint8 crops resident on one device."""

    def __init__(self, arrays, device):
        import torch

        self.device = device
        self.images = [
            torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().to(device, non_blocking=True)
            for array in arrays
        ]
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def _resize(self, image, size: int):
        import torch.nn.functional as F

        return F.interpolate(
            image.unsqueeze(0).float().div_(255.0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def train_batch(self, indices, generator, *, size: int = 224, scale=(0.8, 1.0)):
        """Augmented, normalised float batch for the given crop indices."""
        import torch

        out = []
        for index in indices:
            image = self.images[int(index)]
            height, width = int(image.shape[1]), int(image.shape[2])
            i, j, h, w = random_resized_crop_params(height, width, generator, scale=scale)
            crop = self._resize(image[:, i : i + h, j : j + w], size)
            if torch.rand(1, generator=generator).item() < 0.5:
                crop = crop.flip(-1)
            if torch.rand(1, generator=generator).item() < 0.5:
                crop = crop.flip(-2)
            out.append(crop)
        return (torch.cat(out) - self.mean) / self.std

    def eval_batch(self, indices, *, size: int = 224):
        """Deterministic ``Resize((size, size))`` + normalisation."""
        import torch

        batch = torch.cat([self._resize(self.images[int(index)], size) for index in indices])
        return (batch - self.mean) / self.std
