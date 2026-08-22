"""Balanced parent-identity probe and shuffled-label control."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def balanced_parent_split(
    provenance: pd.DataFrame, seed: int, *, min_crops: int = 4
) -> tuple[pd.DataFrame, float]:
    """Use the same number of train/test crops for every eligible parent.

    Balancing makes the exact chance accuracy `1 / n_parents`, not merely an
    approximation. Extra crops are excluded symmetrically and remain auditable from
    the original provenance table.
    """
    required = {"crop_id", "parent_id", "crop_path"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Probe provenance misses: {missing}")
    sizes = provenance.groupby("parent_id", observed=True).size()
    eligible = sizes[sizes >= min_crops]
    if len(eligible) < 2:
        raise ValueError("Parent probe needs at least two parents with four crops")
    per_partition = int((eligible // 2).min())
    if per_partition < 2:
        raise ValueError("Balanced parent probe requires two crops per partition")
    rng = np.random.default_rng(seed)
    parent_labels = {parent: index for index, parent in enumerate(sorted(eligible.index))}
    records = []
    for parent, label in parent_labels.items():
        group = provenance.loc[provenance.parent_id == parent]
        order = rng.permutation(len(group))[: 2 * per_partition]
        selected = group.iloc[order]
        for local_index, row in enumerate(selected.itertuples(index=False)):
            records.append(
                {
                    "crop_id": row.crop_id,
                    "parent_id": parent,
                    "crop_path": row.crop_path,
                    "label": label,
                    "partition": "train" if local_index < per_partition else "test",
                    "seed": seed,
                }
            )
    out = pd.DataFrame(records).sort_values(["partition", "label", "crop_id"])
    expected = len(parent_labels) * per_partition
    counts = out.groupby(["partition", "label"], observed=True).size()
    if len(out) != 2 * expected or not (counts == per_partition).all():
        raise RuntimeError("Parent probe balancing invariant failed")
    return out.reset_index(drop=True), 1.0 / len(parent_labels)


class _ImageMode:
    def __init__(self, mode: str):
        if mode not in {"rgb", "grayscale", "highpass"}:
            raise ValueError(f"Unsupported image mode: {mode}")
        self.mode = mode

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.mode == "rgb":
            return image
        values = np.asarray(image)
        if self.mode == "grayscale":
            gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)
            values = np.repeat(gray[..., None], 3, axis=2)
        else:
            blurred = cv2.GaussianBlur(values, (0, 0), sigmaX=3)
            values = np.clip(values.astype(np.int16) - blurred.astype(np.int16) + 128, 0, 255)
            values = values.astype(np.uint8)
        return Image.fromarray(values, mode="RGB")


def _git_sha(repository: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_parent_probe(
    provenance: pd.DataFrame,
    *,
    seed: int,
    output: str | Path,
    image_mode: str = "rgb",
    shuffled_labels: bool = False,
    epochs: int = 15,
    batch_size: int = 64,
    workers: int = 2,
    learning_rate: float = 1e-4,
    pretrained: bool = True,
    repository: str | Path = ".",
) -> dict[str, object]:
    """Fine-tune ResNet-18 to identify source parents."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    split, chance = balanced_parent_split(provenance, seed)
    n_parents = split.label.nunique()

    if shuffled_labels:
        rng = np.random.default_rng(seed + 1_000_003)
        train_index = split.index[split.partition == "train"]
        split.loc[train_index, "label"] = rng.permutation(split.loc[train_index, "label"])

    mode = _ImageMode(image_mode)
    train_transform = transforms.Compose(
        [
            mode,
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            mode,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    class ProbeDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform):
            self.frame = frame.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            row = self.frame.iloc[index]
            with Image.open(row.crop_path) as image:
                values = self.transform(image)
            return values, int(row.label)

    train_frame = split.loc[split.partition == "train"]
    test_frame = split.loc[split.partition == "test"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ProbeDataset(train_frame, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )
    test_loader = DataLoader(
        ProbeDataset(test_frame, test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_parents)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    started = time.time()
    history = []
    for epoch in range(epochs):
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(labels)
            train_correct += int((logits.argmax(dim=1) == labels).sum())
            train_total += len(labels)
        model.eval()
        test_correct = test_total = 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                logits = model(images)
                test_correct += int((logits.argmax(dim=1) == labels).sum())
                test_total += len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / train_total,
                "train_accuracy": train_correct / train_total,
                "test_accuracy": test_correct / test_total,
            }
        )

    result = {
        "dataset": "SIPaKMeD",
        "task": "parent_identity_probe",
        "architecture": "resnet18",
        "image_mode": image_mode,
        "shuffled_labels": shuffled_labels,
        "seed": seed,
        "n_parents": n_parents,
        "train_crops": len(train_frame),
        "test_crops": len(test_frame),
        "chance_accuracy": chance,
        "test_accuracy": history[-1]["test_accuracy"],
        "best_test_accuracy": max(item["test_accuracy"] for item in history),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_sha": _git_sha(repository),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def summarize_probe(paths: list[str | Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    from .statistics import mean_sd, shuffled_probe_check

    records = [json.loads(Path(path).read_text()) for path in paths]
    frame = pd.DataFrame(records)
    summary = mean_sd(
        frame,
        value="test_accuracy",
        by=["dataset", "architecture", "image_mode", "shuffled_labels"],
    )
    shuffled = frame.loc[frame.shuffled_labels]
    control = shuffled_probe_check(
        shuffled.test_accuracy.to_numpy(), float(shuffled.chance_accuracy.iloc[0])
    )
    return summary, control

