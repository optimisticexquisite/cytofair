"""Image-level carriers of the parent fingerprint (Study 2 follow-up, Amendment 8).

Every transform keeps the cell's morphology and removes or replaces one candidate
carrier of acquisition context:

* ``background_grey``: everything outside the Study 1 foreground mask becomes the
  ImageNet mean colour (zero after normalisation), so no background pixel survives.
* ``grey``: luminance only, so stain hue and saturation are gone.
* ``stain_transfer``: Reinhard transfer of the crop's LAB mean and standard
  deviation to those of a donor crop from an unseen field of the same class.
* ``low_pass``: Gaussian blur (sigma = 2 px at the 224-px model input) removes
  fine texture, focus, and sensor noise.
* ``illumination_flat``: subtracting a wide Gaussian (sigma = 16 px at 224 px)
  removes slowly varying illumination and vignetting.

``reinhard_normalise`` maps every crop's LAB statistics toward one dataset-wide
reference at a chosen strength; it is used to build training caches for the
stain-normalisation dose experiment.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import pandas as pd

from .image_cache import CropCache, write_array_cache
from .transplant import foreground_mask

# ImageNet mean in uint8, so masked pixels are exactly zero after normalisation.
FILL = (124, 116, 104)
MODEL_SIZE = 224
ABLATIONS = ("original", "grey", "stain_transfer", "low_pass", "illumination_flat", "background_grey")


def _scaled_sigma(values: np.ndarray, sigma_at_model_size: float) -> float:
    return sigma_at_model_size * max(values.shape[:2]) / MODEL_SIZE


def background_grey(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    mask = foreground_mask(values) if mask is None else mask
    out = values.copy()
    out[mask == 0] = FILL
    return out


def grey(values: np.ndarray) -> np.ndarray:
    luminance = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)
    return np.repeat(luminance[..., None], 3, axis=2)


def low_pass(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return cv2.GaussianBlur(values, (0, 0), sigmaX=_scaled_sigma(values, sigma))


def illumination_flat(values: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    pixels = values.astype(np.float32)
    smooth = cv2.GaussianBlur(pixels, (0, 0), sigmaX=_scaled_sigma(values, sigma))
    flat = pixels - smooth + pixels.reshape(-1, 3).mean(axis=0)
    return np.clip(np.rint(flat), 0, 255).astype(np.uint8)


def _to_lab(values: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(values.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def lab_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = _to_lab(values).reshape(-1, 3)
    return lab.mean(axis=0), np.maximum(lab.std(axis=0), 1e-3)


def reinhard(values: np.ndarray, target_mean, target_std, strength: float = 1.0) -> np.ndarray:
    """Move the crop's LAB statistics toward the target by ``strength`` (0 = unchanged)."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if strength == 0.0:
        return values.copy()
    lab = _to_lab(values)
    mean, std = lab.reshape(-1, 3).mean(axis=0), np.maximum(lab.reshape(-1, 3).std(axis=0), 1e-3)
    normalised = (lab - mean) / std * np.asarray(target_std, dtype=np.float32) + np.asarray(target_mean, dtype=np.float32)
    return _from_lab((1.0 - strength) * lab + strength * normalised)


def stain_transfer(values: np.ndarray, donor: np.ndarray) -> np.ndarray:
    mean, std = lab_statistics(donor)
    return reinhard(values, mean, std, 1.0)


def reference_statistics(cache: CropCache, crop_ids) -> tuple[np.ndarray, np.ndarray]:
    """Dataset reference: the mean of per-crop LAB means and of per-crop LAB standard deviations."""
    means, stds = zip(*(lab_statistics(cache.array(crop)) for crop in crop_ids), strict=True)
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def build_masked_cache(cache: CropCache, crop_ids, output_dir: str, name: str) -> tuple[dict, pd.DataFrame]:
    fractions = {}

    def items() -> Iterator[tuple[dict, np.ndarray]]:
        for crop in crop_ids:
            values = cache.array(crop)
            mask = foreground_mask(values)
            fractions[crop] = float(mask.mean())
            yield {"crop_id": crop}, background_grey(values, mask)

    manifest = write_array_cache(
        items(), output_dir, name, source={"kind": "background_grey", "fill": list(FILL), "parent": cache.name}
    )
    return manifest, pd.DataFrame({"crop_id": list(fractions), "foreground_fraction": list(fractions.values())})


def build_reinhard_cache(
    cache: CropCache, crop_ids, output_dir: str, name: str, *, strength: float, reference
) -> dict:
    target_mean, target_std = reference

    def items() -> Iterator[tuple[dict, np.ndarray]]:
        for crop in crop_ids:
            yield {"crop_id": crop}, reinhard(cache.array(crop), target_mean, target_std, strength)

    return write_array_cache(
        items(),
        output_dir,
        name,
        source={
            "kind": "reinhard_normalisation",
            "strength": strength,
            "reference_mean": [float(v) for v in target_mean],
            "reference_std": [float(v) for v in target_std],
            "parent": cache.name,
        },
    )


def ablation_plan(design: pd.DataFrame, transplant_plan: pd.DataFrame) -> pd.DataFrame:
    """One row per (test crop, ablation); the stain donor is the 2B unseen same-class donor."""
    test = sorted(design.loc[design.role == "test"].crop_id)
    donors = transplant_plan.loc[transplant_plan.condition == "unseen_same_class"].set_index("recipient_id").donor_id
    rows = []
    for crop in test:
        for ablation in ABLATIONS:
            rows.append(
                {
                    "ablated_id": f"{crop}|{ablation}",
                    "recipient_id": crop,
                    "ablation": ablation,
                    "donor_id": donors[crop] if ablation == "stain_transfer" else crop,
                }
            )
    return pd.DataFrame(rows)


def build_ablation_cache(plan: pd.DataFrame, cache: CropCache, output_dir: str, name: str) -> dict:
    def items() -> Iterator[tuple[dict, np.ndarray]]:
        for recipient, group in plan.groupby("recipient_id", sort=True):
            values = cache.array(recipient)
            for row in group.itertuples(index=False):
                if row.ablation == "original":
                    pixels = values
                elif row.ablation == "grey":
                    pixels = grey(values)
                elif row.ablation == "stain_transfer":
                    pixels = stain_transfer(values, cache.array(row.donor_id))
                elif row.ablation == "low_pass":
                    pixels = low_pass(values)
                elif row.ablation == "illumination_flat":
                    pixels = illumination_flat(values)
                elif row.ablation == "background_grey":
                    pixels = background_grey(values)
                else:  # pragma: no cover - ABLATIONS is closed
                    raise ValueError(row.ablation)
                yield {"crop_id": row.ablated_id, "recipient_id": recipient, "ablation": row.ablation}, pixels

    return write_array_cache(items(), output_dir, name, source={"kind": "carrier_ablations", "rows": len(plan)})
