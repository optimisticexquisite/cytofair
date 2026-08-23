"""Resumable Cx22 semantic-segmentation harness for matched split protocols."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .classification import classification_split
from .probe import _git_sha, _runtime_metadata

SEMANTIC_ARCHITECTURES = {
    "unet_resnet34": ("Unet", "resnet34"),
    "unetplusplus_resnet34": ("UnetPlusPlus", "resnet34"),
    "deeplabv3plus_resnet50": ("DeepLabV3Plus", "resnet50"),
    "fpn_resnet34": ("FPN", "resnet34"),
    "manet_resnet34": ("MAnet", "resnet34"),
}


def segmentation_split(provenance: pd.DataFrame, protocol: str, seed: int) -> pd.DataFrame:
    if protocol != "OFFICIAL":
        return classification_split(provenance, protocol, seed)
    train = provenance.loc[provenance.subset == "Cx22-Multi-Train"]
    test = provenance.loc[provenance.subset == "Cx22-Multi-Test"]
    if len(train) != 400 or len(test) != 100:
        raise ValueError(f"Cx22 OFFICIAL split must be 400/100, found {len(train)}/{len(test)}")
    out = pd.concat(
        [train.assign(partition="train"), test.assign(partition="test")], ignore_index=True
    )[["crop_id", "parent_id", "class_name", "partition"]]
    train_parents = set(out.loc[out.partition == "train", "parent_id"])
    test_parents = set(out.loc[out.partition == "test", "parent_id"])
    if train_parents & test_parents:
        raise RuntimeError("Cx22 OFFICIAL split unexpectedly shares source parents")
    out["protocol"], out["seed"] = protocol, seed
    return out


def _split_sha256(split: pd.DataFrame) -> str:
    payload = split.sort_values("crop_id")[["crop_id", "partition"]].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _manifest_sha256(provenance: pd.DataFrame) -> str:
    columns = [
        "crop_id",
        "parent_id",
        "image_sha256",
        "cyto_mask_sha256",
        "nuc_mask_sha256",
        "cyto_instances_sha256",
        "cache_images_sha256",
        "cache_semantic_sha256",
        "cache_instances_sha256",
        "training_cache_version",
        "p3_group_id",
    ]
    columns = [column for column in columns if column in provenance]
    payload = provenance.sort_values("crop_id")[columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def _semantic_model(architecture: str, pretrained: bool):
    if architecture not in SEMANTIC_ARCHITECTURES:
        raise ValueError(f"Unknown semantic architecture: {architecture}")
    import segmentation_models_pytorch as smp

    builder_name, encoder = SEMANTIC_ARCHITECTURES[architecture]
    builder = getattr(smp, builder_name)
    return builder(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=2,
        activation=None,
    )


def binary_channel_metrics(probabilities, targets, threshold: float = 0.5):
    """Return per-image/channel Dice and IoU tensors."""
    predictions = probabilities >= threshold
    targets = targets >= 0.5
    axes = (-2, -1)
    intersection = (predictions & targets).sum(dim=axes).float()
    predicted = predictions.sum(dim=axes).float()
    expected = targets.sum(dim=axes).float()
    union = predicted + expected - intersection
    dice = (2 * intersection) / (predicted + expected).clamp_min(1)
    iou = intersection / union.clamp_min(1)
    return dice, iou


def _atomic_torch_save(torch, payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_semantic_segmentation(
    provenance: pd.DataFrame,
    *,
    protocol: str,
    architecture: str,
    seed: int,
    output: str | Path,
    epochs: int = 30,
    batch_size: int = 16,
    workers: int = 2,
    learning_rate: float = 1e-4,
    image_size: int = 256,
    pretrained: bool = True,
    repository: str | Path = ".",
) -> dict[str, object]:
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import functional as vision_functional

    required = {
        "crop_id",
        "parent_id",
        "class_name",
        "subset",
        "crop_path",
        "cyto_mask_path",
        "nuc_mask_path",
    }
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Semantic segmentation provenance misses: {missing}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    split = segmentation_split(provenance, protocol, seed)
    source = provenance.merge(split[["crop_id", "partition"]], on="crop_id", validate="one_to_one")
    train_frame = source.loc[source.partition == "train"].reset_index(drop=True)
    test_frame = source.loc[source.partition == "test"].reset_index(drop=True)

    class Cx22SemanticDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, *, train: bool, epoch: int = 0):
            self.frame, self.train, self.epoch = frame, train, epoch
            self.cached = {
                "cache_index",
                "cache_images_path",
                "cache_semantic_path",
                "cache_image_size",
            }.issubset(frame.columns)
            self.cache_images = None
            self.cache_semantic = None

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            row = self.frame.iloc[index]
            if self.cached:
                if int(row.cache_image_size) != image_size:
                    raise ValueError(
                        f"Cache size {row.cache_image_size} does not match run size {image_size}"
                    )
                if self.cache_images is None:
                    self.cache_images = np.load(row.cache_images_path, mmap_mode="r")
                    self.cache_semantic = np.load(row.cache_semantic_path, mmap_mode="r")
                cache_index = int(row.cache_index)
                image = torch.from_numpy(
                    np.array(self.cache_images[cache_index], copy=True)
                ).permute(2, 0, 1).float() / 255
                target = torch.from_numpy(
                    np.array(self.cache_semantic[cache_index], copy=True)
                ).float()
            else:
                with Image.open(row.crop_path) as image:
                    image = vision_functional.pil_to_tensor(image.convert("RGB")).float() / 255
                channels = []
                for path in (row.cyto_mask_path, row.nuc_mask_path):
                    with Image.open(path) as mask:
                        channels.append(vision_functional.pil_to_tensor(mask.convert("L"))[0])
                target = torch.stack(channels).float() / 255
                image = vision_functional.resize(
                    image, [image_size, image_size], antialias=True
                )
                target = vision_functional.resize(
                    target,
                    [image_size, image_size],
                    interpolation=vision_functional.InterpolationMode.NEAREST,
                )
            if self.train:
                digest = hashlib.sha256(
                    f"{seed}|{self.epoch}|{row.crop_id}".encode()
                ).digest()
                if digest[0] & 1:
                    image, target = image.flip(-1), target.flip(-1)
                if digest[0] & 2:
                    image, target = image.flip(-2), target.flip(-2)
                rotations = digest[1] % 4
                if rotations:
                    image = torch.rot90(image, rotations, dims=(-2, -1))
                    target = torch.rot90(target, rotations, dims=(-2, -1))
            image = vision_functional.normalize(
                image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
            return image, target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    model = _semantic_model(architecture, pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".checkpoint.pt")
    start_epoch, history = 0, []
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = (protocol, architecture, seed, _manifest_sha256(provenance))
        observed = tuple(state[name] for name in ("protocol", "architecture", "seed", "manifest"))
        if observed != expected:
            raise ValueError(f"Refusing incompatible checkpoint {checkpoint}: {observed} != {expected}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, history = int(state["next_epoch"]), list(state["history"])

    test_loader = DataLoader(
        Cx22SemanticDataset(test_frame, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    started, started_at = time.time(), datetime.now(UTC).isoformat()
    for epoch in range(start_epoch, epochs):
        train_loader = DataLoader(
            Cx22SemanticDataset(train_frame, train=True, epoch=epoch),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed + epoch * 1_000_003),
        )
        model.train()
        train_loss, train_total = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(images)
                bce = criterion(logits, targets)
                probabilities = logits.sigmoid()
                intersection = (probabilities * targets).sum(dim=(-2, -1))
                denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
                soft_dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
                loss = bce + soft_dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(targets)
            train_total += len(targets)
        model.eval()
        dice_values, iou_values = [], []
        with torch.inference_mode():
            for images, targets in test_loader:
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    probabilities = model(images.to(device, non_blocking=True)).sigmoid().cpu()
                dice, iou = binary_channel_metrics(probabilities, targets)
                dice_values.append(dice)
                iou_values.append(iou)
        dice = torch.cat(dice_values).numpy()
        iou = torch.cat(iou_values).numpy()
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "macro_dice": float(dice.mean()),
            "cytoplasm_dice": float(dice[:, 0].mean()),
            "nucleus_dice": float(dice[:, 1].mean()),
            "macro_iou": float(iou.mean()),
            "cytoplasm_iou": float(iou[:, 0].mean()),
            "nucleus_iou": float(iou[:, 1].mean()),
        }
        history.append(epoch_record)
        print(json.dumps({"event": "epoch", **epoch_record}), flush=True)
        _atomic_torch_save(
            torch,
            {
                "protocol": protocol,
                "architecture": architecture,
                "seed": seed,
                "manifest": _manifest_sha256(provenance),
                "next_epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
            },
            checkpoint,
        )
        if not np.isfinite(epoch_record["train_loss"]):
            raise FloatingPointError("Non-finite semantic segmentation loss")

    result = {
        "dataset": "Cx22-pooled" if protocol != "OFFICIAL" else "Cx22-official-Multi",
        "task": "semantic_segmentation",
        "architecture": architecture,
        "protocol": protocol,
        "seed": seed,
        **{key: value for key, value in history[-1].items() if key != "epoch"},
        "train_crops": len(train_frame),
        "test_crops": len(test_frame),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "mixed_precision": "fp16_dynamic_loss_scaling" if amp_enabled else "disabled",
        "split_sha256": _split_sha256(split),
        "input_manifest_sha256": _manifest_sha256(provenance),
        "git_sha": _git_sha(repository),
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
        "runtime": _runtime_metadata(torch, torchvision, provenance),
        "history": history,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint.unlink(missing_ok=True)
    return result
