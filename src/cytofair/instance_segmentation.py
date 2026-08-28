"""Resumable Cx22 cytoplasm-instance segmentation and matched-instance metrics."""

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
from scipy.optimize import linear_sum_assignment

from .probe import _git_sha, _runtime_metadata
from .segmentation import _atomic_torch_save, _manifest_sha256, _split_sha256, segmentation_split

INSTANCE_ARCHITECTURES = ("maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2")


def matched_instance_metrics(prediction_masks, scores, target_masks, score_threshold: float = 0.5):
    """Maximum-Dice matching, assigning zero to every unmatched ground-truth instance."""
    prediction_masks = prediction_masks[scores >= score_threshold] >= 0.5
    target_masks = target_masks >= 0.5
    n_targets = len(target_masks)
    if n_targets == 0:
        raise ValueError("Cx22 instance endpoint requires at least one target instance")
    if not len(prediction_masks):
        return np.zeros(n_targets), np.zeros(n_targets)
    predicted = prediction_masks.flatten(1).float()
    expected = target_masks.flatten(1).float()
    intersection = predicted @ expected.T
    predicted_area = predicted.sum(1)[:, None]
    expected_area = expected.sum(1)[None, :]
    dice = (2 * intersection) / (predicted_area + expected_area).clamp_min(1)
    iou = intersection / (predicted_area + expected_area - intersection).clamp_min(1)
    pred_indices, target_indices = linear_sum_assignment(dice.cpu().numpy(), maximize=True)
    matched_dice = np.zeros(n_targets, dtype=float)
    matched_iou = np.zeros(n_targets, dtype=float)
    matched_dice[target_indices] = dice[pred_indices, target_indices].cpu().numpy()
    matched_iou[target_indices] = iou[pred_indices, target_indices].cpu().numpy()
    return matched_dice, matched_iou


def _instance_model(architecture: str, pretrained: bool, image_size: int):
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if architecture == "maskrcnn_resnet50_fpn":
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
        )
    elif architecture == "maskrcnn_resnet50_fpn_v2":
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
        )
    else:
        raise ValueError(f"Unknown instance architecture: {architecture}")
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    return model


def train_instance_segmentation(
    provenance: pd.DataFrame,
    *,
    protocol: str,
    architecture: str,
    seed: int,
    output: str | Path,
    epochs: int = 30,
    batch_size: int = 2,
    workers: int = 2,
    learning_rate: float = 1e-4,
    image_size: int = 256,
    pretrained: bool = True,
    repository: str | Path = ".",
) -> dict[str, object]:
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as vision_functional

    required = {
        "crop_id",
        "parent_id",
        "class_name",
        "subset",
        "crop_path",
        "cyto_instances_path",
    }
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Instance segmentation provenance misses: {missing}")
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

    class Cx22InstanceDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, *, train: bool, epoch: int = 0):
            self.frame, self.train, self.epoch = frame, train, epoch
            self.cached = {
                "cache_index",
                "cache_images_path",
                "cache_instances_path",
                "cache_instance_start",
                "cache_instance_count",
                "cache_image_size",
            }.issubset(frame.columns)
            self.cache_images = None
            self.cache_instances = None

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
                    self.cache_instances = np.load(row.cache_instances_path, mmap_mode="r")
                cache_index = int(row.cache_index)
                start, count = int(row.cache_instance_start), int(row.cache_instance_count)
                image = torch.from_numpy(
                    np.array(self.cache_images[cache_index], copy=True)
                ).permute(2, 0, 1).float() / 255
                masks = torch.from_numpy(
                    np.array(self.cache_instances[start : start + count], copy=True)
                )
            else:
                with Image.open(row.crop_path) as image:
                    image = vision_functional.pil_to_tensor(image.convert("RGB")).float() / 255
                with np.load(row.cyto_instances_path) as archive:
                    masks = torch.from_numpy(archive["masks"].astype(np.uint8))
                image = vision_functional.resize(image, [image_size, image_size], antialias=True)
                masks = vision_functional.resize(
                    masks,
                    [image_size, image_size],
                    interpolation=InterpolationMode.NEAREST,
                )
            if self.train:
                digest = hashlib.sha256(
                    f"{seed}|{self.epoch}|{row.crop_id}".encode()
                ).digest()
                if digest[0] & 1:
                    image, masks = image.flip(-1), masks.flip(-1)
                if digest[0] & 2:
                    image, masks = image.flip(-2), masks.flip(-2)
                rotations = digest[1] % 4
                if rotations:
                    image = torch.rot90(image, rotations, dims=(-2, -1))
                    masks = torch.rot90(masks, rotations, dims=(-2, -1))
            boxes, retained = [], []
            for mask_index, mask in enumerate(masks):
                points = torch.where(mask > 0)
                if not len(points[0]):
                    continue
                y_min, y_max = points[0].min(), points[0].max()
                x_min, x_max = points[1].min(), points[1].max()
                boxes.append(torch.stack((x_min, y_min, x_max + 1, y_max + 1)).float())
                retained.append(mask_index)
            if not retained:
                raise ValueError(f"All instances vanished while resizing {row.crop_id}")
            masks = masks[retained]
            boxes = torch.stack(boxes)
            area = masks.flatten(1).sum(1).float()
            target = {
                "boxes": boxes,
                "labels": torch.ones(len(masks), dtype=torch.int64),
                "masks": masks,
                "image_id": torch.tensor([index]),
                "area": area,
                "iscrowd": torch.zeros(len(masks), dtype=torch.int64),
            }
            return image, target

    def collate(batch):
        return tuple(zip(*batch, strict=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    model = _instance_model(architecture, pretrained, image_size).to(device)
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
        Cx22InstanceDataset(test_frame, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate,
    )
    started, started_at = time.time(), datetime.now(UTC).isoformat()
    for epoch in range(start_epoch, epochs):
        train_loader = DataLoader(
            Cx22InstanceDataset(train_frame, train=True, epoch=epoch),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed + epoch * 1_000_003),
        )
        model.train()
        train_loss, train_total = 0.0, 0
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(images)
            train_total += len(images)
        model.eval()
        image_dice, image_iou, all_dice = [], [], []
        with torch.inference_mode():
            for images, targets in test_loader:
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    predictions = model(
                        [image.to(device, non_blocking=True) for image in images]
                    )
                prediction = predictions[0]
                dice, iou = matched_instance_metrics(
                    prediction["masks"][:, 0].cpu(),
                    prediction["scores"].cpu(),
                    targets[0]["masks"],
                )
                image_dice.append(float(dice.mean()))
                image_iou.append(float(iou.mean()))
                all_dice.extend(dice.tolist())
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "instance_dice": float(np.mean(image_dice)),
            "instance_iou": float(np.mean(image_iou)),
            "recall70": float(np.mean(np.asarray(all_dice) >= 0.70)),
        }
        history.append(epoch_record)
        print(json.dumps({"event": "epoch", **epoch_record}), flush=True)
        if not np.isfinite(epoch_record["train_loss"]):
            raise FloatingPointError("Non-finite instance segmentation loss")
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
    result = {
        "dataset": "Cx22-pooled" if protocol != "OFFICIAL" else "Cx22-official-Multi",
        "task": "instance_segmentation",
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
        "score_threshold": 0.5,
        "match_threshold": 0.70,
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
