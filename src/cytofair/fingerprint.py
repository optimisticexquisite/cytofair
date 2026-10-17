"""Reference parent-fingerprint encoder (Study 1 RGB probe recipe, retained this time).

Study 1 trained the 424-way ResNet-18 parent probe but did not keep its weights.
Study 2 retrains the identical recipe once from the crop cache (with the same
GPU-side augmentation as every Study 2 run), keeps the checkpoint, and embeds
every crop with its penultimate layer. Those embeddings define the secondary
"fingerprint-space" diversity of each training set.
"""

from __future__ import annotations

import io
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .gpu_augment import GpuCrops
from .image_cache import CropCache
from .probe import _git_sha, balanced_parent_split
from .train_study2 import AUGMENTATION_STREAM_OFFSET, _atomic_bytes, _atomic_npz, sha256_file


def train_fingerprint_encoder(
    provenance: pd.DataFrame,
    *,
    cache_dir: str | Path,
    cache_name: str,
    output_dir: str | Path,
    seed: int = 17,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    repository: str | Path = ".",
) -> dict:
    import torch
    from torch import nn
    from torchvision import models

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = CropCache(cache_dir, cache_name, verify=True)
    split, chance = balanced_parent_split(provenance, seed)
    train = split.loc[split.partition == "train"].reset_index(drop=True)
    test = split.loc[split.partition == "test"].reset_index(drop=True)
    train_store = GpuCrops([cache.array(crop) for crop in train.crop_id], device)
    test_store = GpuCrops([cache.array(crop) for crop in test.crop_id], device)
    train_targets = torch.as_tensor(train.label.to_numpy(), device=device)
    test_targets = torch.as_tensor(test.label.to_numpy(), device=device)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, split.label.nunique())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    shuffle_generator = torch.Generator().manual_seed(seed)
    augmentation_generator = torch.Generator().manual_seed(seed + AUGMENTATION_STREAM_OFFSET)
    history, started = [], time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_store), generator=shuffle_generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            images = train_store.train_batch(batch.tolist(), augmentation_generator, scale=(0.85, 1.0))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), train_targets[batch.to(device)])
            loss.backward()
            optimizer.step()
        model.eval()
        correct = torch.zeros((), device=device, dtype=torch.long)
        with torch.inference_mode():
            for start in range(0, len(test_store), batch_size):
                indices = range(start, min(start + batch_size, len(test_store)))
                logits = model(test_store.eval_batch(indices))
                correct += (logits.argmax(1) == test_targets[start : start + len(indices)]).sum()
        history.append({"epoch": epoch + 1, "test_accuracy": float(correct) / len(test)})
        print(json.dumps({"event": "epoch", **history[-1]}), flush=True)
    del train_store, test_store

    # Embed every cached crop with the penultimate layer.
    model.fc = nn.Identity()
    model.eval()
    everything = cache.index.crop_id.tolist()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(everything), 512):
            store = GpuCrops([cache.array(crop) for crop in everything[start : start + 512]], device)
            for inner in range(0, len(store), batch_size):
                chunks.append(model(store.eval_batch(range(inner, min(inner + batch_size, len(store))))).float().cpu())
            del store
    embeddings = torch.cat(chunks).numpy()
    output_dir = Path(output_dir)
    embedding_path = output_dir / "fingerprint_embeddings.npz"
    _atomic_npz(embedding_path, crop_id=np.asarray(everything).astype(str), features=embeddings.astype(np.float16))
    buffer = io.BytesIO()
    torch.save({"architecture": "resnet18", "seed": seed, "state_dict": model.state_dict()}, buffer)
    checkpoint_path = output_dir / "fingerprint_encoder.pt"
    _atomic_bytes(checkpoint_path, buffer.getvalue())
    record = {
        "task": "parent_fingerprint_encoder",
        "seed": seed,
        "n_parents": int(split.label.nunique()),
        "chance": chance,
        "test_accuracy": history[-1]["test_accuracy"],
        "history": history,
        "embedding_path": str(embedding_path),
        "embedding_sha256": sha256_file(embedding_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cache_pixels_sha256": cache.manifest["pixels_sha256"],
        "git_sha": _git_sha(repository),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_bytes(output_dir / "fingerprint_encoder.json", (json.dumps(record, indent=2) + "\n").encode())
    return record
