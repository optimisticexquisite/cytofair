"""Matched classification harness for P1/P1-prime/P2/P3 comparisons."""

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
from sklearn.metrics import balanced_accuracy_score, f1_score

from .probe import _git_sha, _runtime_metadata
from .splits import make_grouped, make_p1, make_p1_prime


def classification_split(provenance: pd.DataFrame, protocol: str, seed: int) -> pd.DataFrame:
    if protocol == "P1":
        return make_p1(provenance, seed)
    if protocol == "P2":
        return make_grouped(provenance, seed, protocol="P2")
    if protocol == "P1-prime":
        return make_p1_prime(make_p1(provenance, seed), make_grouped(provenance, seed), seed)
    if protocol == "P3":
        if "p3_group_id" not in provenance.columns:
            raise ValueError("P3 requires a p3_group_id produced by duplicate-component purging")
        return make_grouped(provenance, seed, protocol="P3", group_column="p3_group_id")
    raise ValueError(f"Unknown protocol: {protocol}")


def _model(architecture: str, n_classes: int, pretrained: bool):
    from torch import nn
    from torchvision import models

    builders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    }
    if architecture not in builders:
        raise ValueError(f"Unknown architecture: {architecture}")
    builder, default_weights = builders[architecture]
    model = builder(weights=default_weights if pretrained else None)
    if architecture.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif architecture == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif architecture in {"efficientnet_b0", "mobilenet_v2"}:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:
        model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
    return model


def train_classifier(
    provenance: pd.DataFrame,
    *,
    dataset: str,
    protocol: str,
    architecture: str,
    seed: int,
    output: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    workers: int = 2,
    learning_rate: float = 1e-4,
    pretrained: bool = True,
    repository: str | Path = ".",
) -> dict[str, object]:
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    required = {"crop_id", "parent_id", "class_name", "crop_path"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Classification provenance misses: {missing}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    split = classification_split(provenance, protocol, seed)
    source = provenance.merge(split[["crop_id", "partition"]], on="crop_id", validate="one_to_one")
    labels = {name: index for index, name in enumerate(sorted(source.class_name.unique()))}
    source["label"] = source.class_name.map(labels)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    class CellDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform):
            self.frame = frame.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            row = self.frame.iloc[index]
            with Image.open(row.crop_path) as image:
                values = self.transform(image.convert("RGB"))
            return values, int(row.label)

    train_frame = source.loc[source.partition == "train"]
    test_frame = source.loc[source.partition == "test"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CellDataset(train_frame, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )
    test_loader = DataLoader(
        CellDataset(test_frame, test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(architecture, len(labels), pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    started = time.time()
    started_at = datetime.now(UTC).isoformat()
    history = []
    final_predictions, final_targets = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(targets)
            train_total += len(targets)
        model.eval()
        predictions, targets_all = [], []
        with torch.inference_mode():
            for images, targets in test_loader:
                logits = model(images.to(device, non_blocking=True))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets_all.extend(targets.tolist())
        accuracy = float(np.mean(np.asarray(predictions) == np.asarray(targets_all)))
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss / train_total, "accuracy": accuracy}
        )
        final_predictions, final_targets = predictions, targets_all

    split_hash = hashlib.sha256(
        split.sort_values("crop_id")[["crop_id", "partition"]].to_csv(index=False).encode()
    ).hexdigest()
    result = {
        "dataset": dataset,
        "task": "classification",
        "architecture": architecture,
        "protocol": protocol,
        "seed": seed,
        "accuracy": history[-1]["accuracy"],
        "macro_f1": f1_score(final_targets, final_predictions, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(final_targets, final_predictions),
        "train_crops": len(train_frame),
        "test_crops": len(test_frame),
        "n_classes": len(labels),
        "class_names": labels,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "split_sha256": split_hash,
        "git_sha": _git_sha(repository),
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
        "runtime": _runtime_metadata(torch, torchvision, provenance),
        "history": history,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result
