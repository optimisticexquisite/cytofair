"""Study 2 trainer: the Study 1 classification recipe with a complete artifact schema.

Compared with :mod:`cytofair.classification`, the optimisation recipe is unchanged
(ImageNet initialisation, AdamW at 1e-4 with weight decay 1e-4, batch 64, 20
epochs, final-epoch endpoint, identical augmentation parameters). What changes is
the evidence each run leaves behind and where the work happens:

* every split is read from a released CSV whose SHA-256 is checked before use;
* inputs come from a checksummed crop cache, are uploaded to the GPU once, and
  are augmented there (the training host allows four CPU cores per user);
* per-crop test logits, a final checkpoint, penultimate features for the
  balanced parent-probe crops, and logits on optional extra evaluation sets
  (counterfactual composites) are written next to the result JSON;
* an optional class-conditional parent adversary behind a gradient-reversal
  layer implements the fingerprint-suppression arm (2A).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from .backbones import build_classifier, build_parent_adversary, dann_schedule, gradient_reversal
from .gpu_augment import IMAGENET_MEAN, IMAGENET_STD, GpuCrops
from .image_cache import CropCache
from .probe import _git_sha, _runtime_metadata

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "train_study2"]

AUGMENTATION_STREAM_OFFSET = 7919


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_split_sha256(split: pd.DataFrame) -> str:
    """Hash of (crop_id, partition) only, comparable with Study 1 split hashes."""
    payload = split.sort_values("crop_id")[["crop_id", "partition"]].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_split(path: str | Path, expected_sha256: str | None = None) -> pd.DataFrame:
    path = Path(path)
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"Split {path} has SHA-256 {observed}, expected {expected_sha256}")
    split = pd.read_csv(path)
    required = {"crop_id", "parent_id", "class_name", "partition"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Split {path} misses columns: {missing}")
    split = split.loc[split.partition.isin(["train", "test"])].reset_index(drop=True)
    if not split.crop_id.is_unique:
        raise ValueError(f"Split {path} repeats crop IDs")
    if not {"train", "test"} <= set(split.partition):
        raise ValueError(f"Split {path} needs both train and test rows")
    return split


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_bytes(path, buffer.getvalue())


def _optional_version(module: str) -> str | None:
    try:
        imported = __import__(module)
    except ImportError:
        return None
    return getattr(imported, "__version__", None)


def train_study2(
    *,
    cache_dir: str | Path,
    cache_name: str,
    split_path: str | Path,
    dataset: str,
    protocol: str,
    architecture: str,
    seed: int,
    output_dir: str | Path,
    run_name: str,
    split_sha256: str | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    adversary_lambda: float | None = None,
    feature_split_path: str | Path | None = None,
    extra_eval: list[tuple[str, str | Path, str]] | None = None,
    save_checkpoint: bool = True,
    pretrained: bool = True,
    repository: str | Path = ".",
) -> dict[str, object]:
    import torch
    import torchvision
    from torch import nn

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Fused flash / memory-efficient / cuDNN attention kernels have
    # non-deterministic backward passes; the math kernel is deterministic and the
    # token counts here (at most 257) keep its cost modest.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    output_dir = Path(output_dir)
    cache = CropCache(cache_dir, cache_name, verify=True)
    split = load_split(split_path, split_sha256)
    absent = sorted(set(split.crop_id) - set(cache.index.crop_id))
    if absent:
        raise ValueError(f"{len(absent)} split crops are absent from cache {cache_name}: {absent[:3]}")
    class_names = sorted(split.class_name.unique())
    labels = {name: index for index, name in enumerate(class_names)}
    split["label"] = split.class_name.map(labels).astype(int)
    train_frame = split.loc[split.partition == "train"].reset_index(drop=True)
    test_frame = split.loc[split.partition == "test"].reset_index(drop=True)
    train_parents = sorted(train_frame.parent_id.unique())
    parent_index = {parent: index for index, parent in enumerate(train_parents)}
    train_frame["parent_label"] = train_frame.parent_id.map(parent_index).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_store = GpuCrops([cache.array(crop) for crop in train_frame.crop_id], device)
    test_store = GpuCrops([cache.array(crop) for crop in test_frame.crop_id], device)
    train_targets = torch.as_tensor(train_frame.label.to_numpy(copy=True), device=device)
    train_parent_targets = torch.as_tensor(train_frame.parent_label.to_numpy(copy=True), device=device)
    test_targets = test_frame.label.to_numpy(copy=True)
    test_inputs = torch.cat(
        [test_store.eval_batch(range(start, min(start + batch_size, len(test_store))))
         for start in range(0, len(test_store), batch_size)]
    )
    del test_store

    model, spec = build_classifier(architecture, len(class_names), pretrained=pretrained)
    model = model.to(device)
    parameters = list(model.parameters())
    adversary = None
    parent_mask = None
    if adversary_lambda is not None:
        if adversary_lambda < 0:
            raise ValueError("adversary_lambda must be non-negative")
        adversary = build_parent_adversary(spec.feature_dim, len(train_parents)).to(device)
        parameters += list(adversary.parameters())
        # Class-conditional adversary: only parents of the sample's own class
        # compete, so suppressing parent identity never requires removing class
        # information that the classifier legitimately needs.
        parent_class = train_frame.drop_duplicates("parent_id").set_index("parent_id").label
        allowed = np.zeros((len(class_names), len(train_parents)), dtype=bool)
        for parent, index in parent_index.items():
            allowed[int(parent_class[parent]), index] = True
        parent_mask = torch.from_numpy(allowed).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    def evaluate_tensor(inputs):
        model.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                chunks.append(model(inputs[start : start + batch_size]).float())
        return torch.cat(chunks).cpu().numpy()

    shuffle_generator = torch.Generator().manual_seed(seed)
    augmentation_generator = torch.Generator().manual_seed(seed + AUGMENTATION_STREAM_OFFSET)
    n_train = len(train_store)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    step = 0
    history = []
    started = time.time()
    started_at = datetime.now(UTC).isoformat()
    for epoch in range(epochs):
        model.train()
        if adversary is not None:
            adversary.train()
        loss_sum = torch.zeros((), device=device)
        adv_loss_sum = torch.zeros((), device=device)
        adv_correct = torch.zeros((), device=device, dtype=torch.long)
        coefficient = 0.0
        order = torch.randperm(n_train, generator=shuffle_generator)
        for start in range(0, n_train, batch_size):
            batch = order[start : start + batch_size]
            images = train_store.train_batch(batch.tolist(), augmentation_generator)
            batch_device = batch.to(device)
            targets = train_targets[batch_device]
            optimizer.zero_grad(set_to_none=True)
            features = model.features(images)
            loss = criterion(model.head(features), targets)
            total = loss
            if adversary is not None:
                coefficient = dann_schedule(step / max(1, total_steps), adversary_lambda)
                parent_logits = adversary(gradient_reversal(features, coefficient))
                parent_logits = parent_logits.masked_fill(~parent_mask[targets], float("-inf"))
                parents = train_parent_targets[batch_device]
                adv_loss = criterion(parent_logits, parents)
                total = loss + adv_loss
                adv_loss_sum += adv_loss.detach() * len(batch)
                adv_correct += (parent_logits.argmax(dim=1) == parents).sum()
            total.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(batch)
            step += 1
        train_loss = float(loss_sum) / n_train
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
        epoch_logits = evaluate_tensor(test_inputs)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "accuracy": float((epoch_logits.argmax(axis=1) == test_targets).mean()),
        }
        if adversary is not None:
            record.update(
                adversary_loss=float(adv_loss_sum) / n_train,
                adversary_within_class_accuracy=float(adv_correct) / n_train,
                grl_coefficient=coefficient,
            )
        history.append(record)
        print(json.dumps({"event": "epoch", **record}), flush=True)
    training_seconds = time.time() - started
    del train_store

    logits = epoch_logits
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    nll = float(-np.log(np.clip(probabilities[np.arange(len(test_targets)), test_targets], 1e-12, None)).mean())

    artifacts = {}
    prediction_frame = pd.DataFrame(
        {
            "crop_id": test_frame.crop_id,
            "parent_id": test_frame.parent_id,
            "label": test_targets,
            "prediction": predictions,
            **{f"logit_{index}": logits[:, index] for index in range(len(class_names))},
        }
    )
    prediction_path = output_dir / "predictions" / f"{run_name}.parquet"
    buffer = io.BytesIO()
    prediction_frame.to_parquet(buffer, index=False)
    _atomic_bytes(prediction_path, buffer.getvalue())
    artifacts["predictions"] = {"path": str(prediction_path), "sha256": sha256_file(prediction_path)}

    def run_on_cache(source: CropCache, crop_ids, *, features: bool) -> np.ndarray:
        model.eval()
        out = []
        crop_ids = list(crop_ids)
        with torch.inference_mode():
            for start in range(0, len(crop_ids), 512):
                store = GpuCrops([source.array(crop) for crop in crop_ids[start : start + 512]], device)
                for inner in range(0, len(store), batch_size):
                    inputs = store.eval_batch(range(inner, min(inner + batch_size, len(store))))
                    values = model.features(inputs) if features else model(inputs)
                    out.append(values.float().cpu())
                del store
        return torch.cat(out).numpy()

    if feature_split_path is not None:
        probe = pd.read_csv(feature_split_path)
        features = run_on_cache(cache, probe.crop_id, features=True)
        feature_path = output_dir / "features" / f"{run_name}.npz"
        _atomic_npz(
            feature_path,
            crop_id=probe.crop_id.to_numpy().astype(str),
            features=features.astype(np.float16),
        )
        artifacts["probe_features"] = {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "split_sha256": sha256_file(feature_split_path),
        }

    for name, extra_dir, extra_name in extra_eval or []:
        extra_cache = CropCache(extra_dir, extra_name, verify=True)
        crop_ids = extra_cache.index.crop_id.tolist()
        extra_logits = run_on_cache(extra_cache, crop_ids, features=False)
        extra_path = output_dir / "extra" / f"{run_name}.{name}.npz"
        _atomic_npz(extra_path, crop_id=np.asarray(crop_ids).astype(str), logits=extra_logits.astype(np.float32))
        artifacts[f"extra:{name}"] = {
            "path": str(extra_path),
            "sha256": sha256_file(extra_path),
            "cache_pixels_sha256": extra_cache.manifest["pixels_sha256"],
        }

    if save_checkpoint:
        state = {
            key: value.half() if value.is_floating_point() else value
            for key, value in model.state_dict().items()
        }
        buffer = io.BytesIO()
        torch.save(
            {
                "architecture": architecture,
                "class_names": class_names,
                "seed": seed,
                "protocol": protocol,
                "dataset": dataset,
                "precision": "float16-weights",
                "state_dict": state,
            },
            buffer,
        )
        checkpoint_path = output_dir / "checkpoints" / f"{run_name}.pt"
        _atomic_bytes(checkpoint_path, buffer.getvalue())
        artifacts["checkpoint"] = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}

    leaked = test_frame.parent_id.isin(set(train_parents))
    result = {
        "study": "study2",
        "dataset": dataset,
        "task": "classification",
        "architecture": architecture,
        "protocol": protocol,
        "seed": seed,
        "accuracy": float((predictions == test_targets).mean()),
        "macro_f1": float(f1_score(test_targets, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(test_targets, predictions)),
        "nll": nll,
        "ece_15": expected_calibration_error(probabilities, test_targets),
        "confusion_matrix": confusion_matrix(
            test_targets, predictions, labels=list(range(len(class_names)))
        ).tolist(),
        "class_names": class_names,
        "train_crops": len(train_frame),
        "test_crops": len(test_frame),
        "train_parents": len(train_parents),
        "test_parents": int(test_frame.parent_id.nunique()),
        "test_crops_with_train_parent_fraction": float(leaked.mean()),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "augmentation": {
            "device": "gpu",
            "random_resized_crop": {"size": 224, "scale": [0.8, 1.0], "ratio": [0.75, 4 / 3]},
            "horizontal_flip": 0.5,
            "vertical_flip": 0.5,
            "interpolation": "bilinear_antialias",
            "shuffle_seed": seed,
            "augmentation_seed": seed + AUGMENTATION_STREAM_OFFSET,
        },
        "pretrained": pretrained,
        "weights": spec.weights,
        "feature_dim": spec.feature_dim,
        "adversary_lambda": adversary_lambda,
        "adversary": None
        if adversary is None
        else {
            "kind": "class_conditional_parent_head_behind_gradient_reversal",
            "schedule": "dann_ramp",
            "hidden": 512,
            "n_parents": len(train_parents),
        },
        "split_path": str(split_path),
        "split_file_sha256": sha256_file(split_path),
        "split_sha256": semantic_split_sha256(split),
        "cache_name": cache_name,
        "cache_pixels_sha256": cache.manifest["pixels_sha256"],
        "cache_index_sha256": cache.manifest["index_sha256"],
        "deterministic_algorithms": "warn_only",
        "attention_kernels": "math_sdp_only",
        "git_sha": _git_sha(repository),
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "training_seconds": training_seconds,
        "elapsed_seconds": time.time() - started,
        "runtime": {
            **_runtime_metadata(torch, torchvision, split),
            "timm_version": _optional_version("timm"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
        "artifacts": artifacts,
        "history": history,
    }
    result_path = output_dir / "runs" / f"{run_name}.json"
    _atomic_bytes(result_path, (json.dumps(result, indent=2) + "\n").encode())
    return result
