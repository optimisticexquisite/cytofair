"""Evaluation-only follow-up analyses on saved Study 2 checkpoints (Amendment 8).

* :func:`factorial_followup` evaluates one 2C model on the carrier ablations of
  its common test set and, for convolutional backbones, measures how much
  Grad-CAM attribution falls outside the cell mask.
* :func:`leace_followup` erases within-class parent identity from one Phase 1
  model's penultimate features at a ladder of ranks, refits the linear class
  head on the erased features, and measures test accuracy and parent
  recoverability at every rank.

Both verify the checkpoint hash recorded by the training run before loading it.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .image_cache import CropCache

LEACE_RANKS = (0, 4, 16, 64, 256)
CAM_LAYERS = {
    "resnet50": lambda model: model.backbone.layer4,
    "densenet121": lambda model: model.backbone.features,
    "convnext_tiny": lambda model: model.backbone.features,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _setup_torch(seed: int):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(run: dict, device):
    import torch

    from .backbones import build_classifier

    artifact = run["artifacts"]["checkpoint"]
    if _sha256(artifact["path"]) != artifact["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {artifact['path']}")
    checkpoint = torch.load(artifact["path"], map_location="cpu", weights_only=False)
    model, _ = build_classifier(checkpoint["architecture"], len(checkpoint["class_names"]), pretrained=False)
    state = {key: value.float() if value.is_floating_point() else value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint


def run_cache(model, cache: CropCache, crop_ids, device, *, features: bool, batch_size: int = 128):
    """Logits or penultimate features for the given crops (eval resize, no augmentation)."""
    import torch

    from .gpu_augment import GpuCrops

    crop_ids = list(crop_ids)
    out = []
    with torch.inference_mode():
        for start in range(0, len(crop_ids), 512):
            store = GpuCrops([cache.array(crop) for crop in crop_ids[start : start + 512]], device)
            for inner in range(0, len(store), batch_size):
                inputs = store.eval_batch(range(inner, min(inner + batch_size, len(store))))
                values = model.features(inputs) if features else model(inputs)
                out.append(values.float())
            del store
    # Clone outside inference mode so later fits can use the values in autograd.
    return torch.cat(out).clone()


def fit_logistic(x, y, n_classes: int, *, C: float = 1.0, max_iter: int = 1000):
    """Multinomial logistic regression on standardised features (sklearn's objective, L-BFGS on device).

    Minimises ``0.5 * ||W||^2 + C * sum(cross-entropy)`` with an unpenalised
    intercept, as ``sklearn.linear_model.LogisticRegression(C=C)`` does after a
    ``StandardScaler``. Returns a function mapping raw features to logits.
    """
    import torch
    import torch.nn.functional as F

    x = x.float()
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    z = (x - mean) / std
    weight = torch.zeros(x.shape[1], n_classes, device=x.device, requires_grad=True)
    bias = torch.zeros(n_classes, device=x.device, requires_grad=True)
    n = len(y)
    optimizer = torch.optim.LBFGS(
        [weight, bias], lr=1.0, max_iter=max_iter, tolerance_grad=1e-6, tolerance_change=1e-10,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = (C * F.cross_entropy(z @ weight + bias, y, reduction="sum") + 0.5 * weight.pow(2).sum()) / n
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    weight, bias = weight.detach(), bias.detach()
    return lambda values: ((values.float() - mean) / std) @ weight + bias


def _group_sums(values, codes, n_groups: int):
    """Per-group sums as a one-hot product (deterministic on CUDA, unlike ``index_add_``)."""
    import torch

    onehot = torch.zeros(n_groups, len(codes), dtype=values.dtype, device=values.device)
    onehot[codes, torch.arange(len(codes), device=values.device)] = 1.0
    return onehot @ values


def within_class_recoverability(features, probe: pd.DataFrame, crop_index: dict[str, int]) -> dict:
    """Probe-split parent identification (logistic and nearest-centroid), within class."""
    import torch

    parents = sorted(probe.parent_id.unique())
    parent_index = {parent: index for index, parent in enumerate(parents)}
    parent_class = probe.drop_duplicates("parent_id").set_index("parent_id").class_name
    class_of = np.asarray([parent_class[parent] for parent in parents])
    rows = torch.as_tensor([crop_index[crop] for crop in probe.crop_id], device=features.device)
    values = features[rows]
    y = torch.as_tensor(probe.parent_id.map(parent_index).to_numpy(copy=True), device=features.device)
    fit = torch.as_tensor((probe.probe_partition == "fit").to_numpy(copy=True), device=features.device)
    evaluate = ~fit
    eval_class = probe.class_name.to_numpy()[(probe.probe_partition == "eval").to_numpy()]
    same = torch.as_tensor(class_of[None, :] == eval_class[:, None], device=features.device)
    predict = fit_logistic(values[fit], y[fit], len(parents))
    logits = predict(values[evaluate])
    truth = y[evaluate]
    normed = values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)
    centroids = _group_sums(normed[fit], y[fit], len(parents))
    centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)
    similarity = normed[evaluate] @ centroids.T
    counts = pd.Series(class_of).value_counts()
    return {
        "within_class_accuracy": float((logits.masked_fill(~same, float("-inf")).argmax(1) == truth).float().mean()),
        "overall_accuracy": float((logits.argmax(1) == truth).float().mean()),
        "within_class_ncc_accuracy": float(
            (similarity.masked_fill(~same, float("-inf")).argmax(1) == truth).float().mean()
        ),
        "within_class_floor": float(np.mean([1.0 / counts[c] for c in eval_class])),
        "chance": 1.0 / len(parents),
    }


def within_class_eraser(x, parent_codes, class_codes, ranks, *, ridge: float = 1e-6):
    """Rank-limited LEACE erasure of within-class parent identity.

    With ``W = Sigma^{-1/2}`` and the within-class between-parent scatter
    ``S = sum_j (n_j / n)(mu_j - mu_c(j))(mu_j - mu_c(j))^T``, the eraser removes the
    top-``r`` eigendirections ``U_r`` of ``W S W``:
    ``x' = x - (x - mu) W U_r U_r^T W^{-1}``. At full rank this is LEACE (Belrose et
    al., 2023) for the within-class parent indicator, so no linear function of the
    erased features is correlated with it on the fitting data.
    Returns ``(mean, {rank: matrix or None}, eigenvalues)``.
    """
    import torch

    data = x.double()
    n, dimension = data.shape
    mean = data.mean(dim=0)
    centred = data - mean
    covariance = centred.T @ centred / n
    covariance = covariance + ridge * torch.trace(covariance) / dimension * torch.eye(
        dimension, dtype=covariance.dtype, device=covariance.device
    )
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(values.max() * 1e-10)
    whiten = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    unwhiten = vectors @ torch.diag(values.sqrt()) @ vectors.T

    n_parents = int(parent_codes.max()) + 1
    counts = _group_sums(torch.ones(n, 1, dtype=data.dtype, device=data.device), parent_codes, n_parents)[:, 0]
    parent_means = _group_sums(data, parent_codes, n_parents) / counts[:, None]
    n_classes = int(class_codes.max()) + 1
    class_counts = _group_sums(torch.ones(n, 1, dtype=data.dtype, device=data.device), class_codes, n_classes)[:, 0]
    class_means = _group_sums(data, class_codes, n_classes) / class_counts[:, None]
    parent_class = torch.zeros(n_parents, dtype=torch.long, device=data.device)
    parent_class[parent_codes] = class_codes
    differences = parent_means - class_means[parent_class]
    scatter = (differences * (counts / n)[:, None]).T @ differences
    whitened = whiten @ scatter @ whiten
    eigenvalues, eigenvectors = torch.linalg.eigh((whitened + whitened.T) / 2)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    erasers = {}
    for rank in ranks:
        if rank == 0:
            erasers[rank] = None
            continue
        basis = eigenvectors[:, :rank]
        erasers[rank] = (whiten @ basis @ basis.T @ unwhiten).float()
    return mean.float(), erasers, eigenvalues


def apply_eraser(values, mean, matrix):
    return values if matrix is None else values - (values - mean) @ matrix


def leace_followup(
    run_json: str | Path,
    cache_dir: str | Path,
    cache_name: str,
    probe_path: str | Path,
    output_dir: str | Path,
    *,
    ranks=LEACE_RANKS,
    repository: str | Path = ".",
) -> dict:
    from .probe import _git_sha
    from .train_study2 import load_split

    run = json.loads(Path(run_json).read_text())
    torch, device = _setup_torch(int(run["seed"]))
    started = time.time()
    model, checkpoint = load_checkpoint_model(run, device)
    cache = CropCache(cache_dir, cache_name, verify=True)
    if cache.manifest["pixels_sha256"] != run["cache_pixels_sha256"]:
        raise ValueError("cache differs from the one the model was trained on")
    split = load_split(run["split_path"], run["split_file_sha256"])
    probe = pd.read_csv(probe_path)
    crops = sorted(set(split.crop_id) | set(probe.crop_id))
    crop_index = {crop: index for index, crop in enumerate(crops)}
    features = run_cache(model, cache, crops, device, features=True)
    class_names = checkpoint["class_names"]
    labels = {name: index for index, name in enumerate(class_names)}
    train = split.loc[split.partition == "train"]
    test = split.loc[split.partition == "test"]
    train_rows = torch.as_tensor([crop_index[c] for c in train.crop_id], device=device)
    test_rows = torch.as_tensor([crop_index[c] for c in test.crop_id], device=device)
    y_train = torch.as_tensor(train.class_name.map(labels).to_numpy(copy=True), device=device)
    y_test = torch.as_tensor(test.class_name.map(labels).to_numpy(copy=True), device=device)
    parents = {parent: index for index, parent in enumerate(sorted(train.parent_id.unique()))}
    parent_codes = torch.as_tensor(train.parent_id.map(parents).to_numpy(copy=True), device=device)

    with torch.inference_mode():
        original_head = model.head(features[test_rows]).argmax(1)
    original_accuracy = float((original_head == y_test).float().mean())
    mean, erasers, spectrum = within_class_eraser(features[train_rows], parent_codes, y_train, ranks)
    results = []
    for rank in ranks:
        erased = apply_eraser(features, mean, erasers[rank])
        predict = fit_logistic(erased[train_rows], y_train, len(class_names))
        accuracy = float((predict(erased[test_rows]).argmax(1) == y_test).float().mean())
        results.append({"rank": int(rank), "accuracy": accuracy, **within_class_recoverability(erased, probe, crop_index)})
        print(json.dumps({"event": "rank", **results[-1]}), flush=True)

    output_dir = Path(output_dir)
    name = Path(run_json).stem
    spectrum_path = output_dir / "spectra" / f"{name}.npy"
    buffer = io.BytesIO()
    np.save(buffer, spectrum[:512].cpu().numpy())
    _atomic_write(spectrum_path, buffer.getvalue())
    record = {
        "study": "study2-followup",
        "experiment": "leace",
        "source_run": str(run_json),
        "dataset": run["dataset"],
        "protocol": run["protocol"],
        "architecture": run["architecture"],
        "seed": run["seed"],
        "split_file_sha256": run["split_file_sha256"],
        "feature_dim": int(features.shape[1]),
        "train_parents": len(parents),
        "original_head_accuracy": original_accuracy,
        "recorded_accuracy": run["accuracy"],
        "ranks": results,
        "probe_split_sha256": _sha256(probe_path),
        "git_sha": _git_sha(repository),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
        "artifacts": {"spectrum": {"path": str(spectrum_path), "sha256": _sha256(spectrum_path)}},
    }
    _atomic_write(output_dir / "runs" / f"{name}.json", (json.dumps(record, indent=2) + "\n").encode())
    return record


def grad_cam(model, layer, inputs, targets):
    """Grad-CAM (Selvaraju et al., 2017) of the target-class logit at ``layer``."""
    import torch
    import torch.nn.functional as F

    store = {}
    handle = layer.register_forward_hook(lambda module, args, output: store.__setitem__("value", output))
    try:
        with torch.enable_grad():
            logits = model(inputs)
            activation = store["value"]
            score = logits.gather(1, targets[:, None]).sum()
            gradient = torch.autograd.grad(score, activation)[0]
    finally:
        handle.remove()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    return F.relu((weights * activation).sum(dim=1)).detach()


def _spearman_rows(a, b):

    ranks_a = a.argsort(dim=1).argsort(dim=1).float()
    ranks_b = b.argsort(dim=1).argsort(dim=1).float()
    ranks_a = ranks_a - ranks_a.mean(dim=1, keepdim=True)
    ranks_b = ranks_b - ranks_b.mean(dim=1, keepdim=True)
    return (ranks_a * ranks_b).sum(1) / (ranks_a.norm(dim=1) * ranks_b.norm(dim=1)).clamp_min(1e-12)


def factorial_followup(
    run_json: str | Path,
    cache_dir: str | Path,
    cache_name: str,
    ablation_dir: str | Path,
    ablation_name: str,
    output_dir: str | Path,
    *,
    repository: str | Path = ".",
) -> dict:
    import torch.nn.functional as F

    from .probe import _git_sha
    from .train_study2 import load_split
    from .transplant import foreground_mask

    run = json.loads(Path(run_json).read_text())
    torch, device = _setup_torch(int(run["seed"]))
    started = time.time()
    model, checkpoint = load_checkpoint_model(run, device)
    ablations = CropCache(ablation_dir, ablation_name, verify=True)
    ablated_ids = ablations.index.crop_id.tolist()
    logits = run_cache(model, ablations, ablated_ids, device, features=False).cpu().numpy()
    output_dir = Path(output_dir)
    name = Path(run_json).stem
    logits_path = output_dir / "logits" / f"{name}.npz"
    _atomic_write(logits_path, _npz_bytes(crop_id=np.asarray(ablated_ids).astype(str), logits=logits.astype(np.float32)))
    artifacts = {"ablation_logits": {"path": str(logits_path), "sha256": _sha256(logits_path)}}

    split = load_split(run["split_path"], run["split_file_sha256"])
    test = split.loc[split.partition == "test"].sort_values("crop_id").reset_index(drop=True)
    labels = {label: index for index, label in enumerate(checkpoint["class_names"])}
    cam_summary = None
    if run["architecture"] in CAM_LAYERS:
        from .gpu_augment import GpuCrops

        cache = CropCache(cache_dir, cache_name, verify=True)
        layer = CAM_LAYERS[run["architecture"]](model)
        random_head = torch.nn.Linear(model.head.in_features, model.head.out_features).to(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261019)
            reference = torch.nn.Linear(model.head.in_features, model.head.out_features)
        random_head.load_state_dict(reference.state_dict())
        trained_head = model.head
        rows = []
        for start in range(0, len(test), 32):
            chunk = test.iloc[start : start + 32]
            arrays = [cache.array(crop) for crop in chunk.crop_id]
            inputs = GpuCrops(arrays, device).eval_batch(range(len(arrays)))
            targets = torch.as_tensor(chunk.class_name.map(labels).to_numpy(copy=True), device=device)
            cam = grad_cam(model, layer, inputs, targets)
            model.head = random_head
            try:
                cam_random = grad_cam(model, layer, inputs, targets)
            finally:
                model.head = trained_head
            agreement = _spearman_rows(cam.flatten(1), cam_random.flatten(1)).cpu().numpy()
            for position, (crop, values) in enumerate(zip(chunk.crop_id, arrays, strict=True)):
                mask = torch.as_tensor(foreground_mask(values), dtype=torch.float32, device=device)
                upsampled = F.interpolate(cam[position][None, None], size=tuple(mask.shape), mode="bilinear",
                                          align_corners=False)[0, 0]
                total = float(upsampled.sum())
                rows.append({
                    "crop_id": crop,
                    "background_pixel_fraction": float(1.0 - mask.mean()),
                    "background_attribution_fraction": float((upsampled * (1.0 - mask)).sum()) / total
                    if total > 0 else float("nan"),
                    "random_head_spearman": float(agreement[position]),
                })
        cams = pd.DataFrame(rows)
        cam_path = output_dir / "gradcam" / f"{name}.parquet"
        buffer = io.BytesIO()
        cams.to_parquet(buffer, index=False)
        _atomic_write(cam_path, buffer.getvalue())
        artifacts["gradcam"] = {"path": str(cam_path), "sha256": _sha256(cam_path)}
        cam_summary = {
            "crops": int(cams.background_attribution_fraction.notna().sum()),
            "mean_background_attribution_fraction": float(cams.background_attribution_fraction.mean()),
            "mean_background_pixel_fraction": float(cams.background_pixel_fraction.mean()),
            "median_random_head_spearman": float(cams.random_head_spearman.median()),
        }

    record = {
        "study": "study2-followup",
        "experiment": "factorial_followup",
        "source_run": str(run_json),
        "dataset": run["dataset"],
        "protocol": run["protocol"],
        "architecture": run["architecture"],
        "seed": run["seed"],
        "split_file_sha256": run["split_file_sha256"],
        "ablation_cache_pixels_sha256": ablations.manifest["pixels_sha256"],
        "gradcam": cam_summary,
        "git_sha": _git_sha(repository),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
        "artifacts": artifacts,
    }
    _atomic_write(output_dir / "runs" / f"{name}.json", (json.dumps(record, indent=2) + "\n").encode())
    return record
