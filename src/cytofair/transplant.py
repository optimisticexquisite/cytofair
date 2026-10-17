"""Counterfactual background transplants for the common 2C test set (Study 2, arm 2B).

The cell foreground of each test crop is kept and composited onto the
inpainted (cell-removed) background of a donor crop. Foreground masks and
inpainting reuse the Study 1 background-only probe recipe exactly: Otsu
thresholds on HSV saturation and grayscale luminance, a 5x5 morphological
closing, and Telea inpainting with radius 5. The feathered alpha mask is the
same for every condition, so seam artifacts are matched across conditions and
are measured directly by the ``self`` condition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import cv2
import numpy as np
import pandas as pd

from .image_cache import CropCache, write_array_cache

CONDITIONS = ("original", "self", "sibling", "unseen_same_class", "unseen_other_class")
FEATHER_SIGMA = 2.0


def foreground_mask(values: np.ndarray) -> np.ndarray:
    """Binary cell mask, identical to the Study 1 ``background_only`` probe mode."""
    hsv = cv2.cvtColor(values, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)
    saturation_threshold, _ = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    luminance_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((hsv[..., 1] > saturation_threshold) | (gray < luminance_threshold)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))


def cell_removed_background(values: np.ndarray) -> np.ndarray:
    return cv2.inpaint(values, foreground_mask(values), 5, cv2.INPAINT_TELEA)


def composite(recipient: np.ndarray, donor_background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Feathered paste of the recipient foreground onto a resized donor background."""
    height, width = recipient.shape[:2]
    background = cv2.resize(donor_background, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=FEATHER_SIGMA)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    mixed = alpha * recipient.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _choice(options: list[str], *key: object) -> str:
    if not options:
        raise ValueError(f"No donor available for {key}")
    ranked = sorted(
        options, key=lambda value: hashlib.sha256("|".join(map(str, (*key, value))).encode()).hexdigest()
    )
    return ranked[0]


def transplant_plan(design: pd.DataFrame) -> pd.DataFrame:
    """Deterministic donor assignment for every test crop and condition."""
    required = {"crop_id", "parent_id", "class_name", "role", "seed"}
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"Transplant design misses: {missing}")
    seed = int(design.seed.iloc[0])
    siblings = design.loc[design.role == "donor_sibling"].groupby("parent_id").crop_id.apply(sorted)
    unseen = design.loc[design.role == "unseen_background"]
    unseen_by_class = unseen.groupby("class_name").crop_id.apply(sorted)
    parent_of = design.set_index("crop_id").parent_id
    classes = sorted(design.class_name.unique())
    rows = []
    for recipient in design.loc[design.role == "test"].sort_values("crop_id").itertuples(index=False):
        base = {
            "recipient_id": recipient.crop_id,
            "recipient_parent": recipient.parent_id,
            "recipient_class": recipient.class_name,
        }
        rows.append({**base, "condition": "original", "donor_id": recipient.crop_id, "donor_class": recipient.class_name})
        rows.append({**base, "condition": "self", "donor_id": recipient.crop_id, "donor_class": recipient.class_name})
        sibling = _choice(list(siblings[recipient.parent_id]), seed, recipient.crop_id, "sibling")
        rows.append({**base, "condition": "sibling", "donor_id": sibling, "donor_class": recipient.class_name})
        same = _choice(list(unseen_by_class[recipient.class_name]), seed, recipient.crop_id, "same")
        rows.append(
            {**base, "condition": "unseen_same_class", "donor_id": same, "donor_class": recipient.class_name}
        )
        for other in classes:
            if other == recipient.class_name:
                continue
            donor = _choice(list(unseen_by_class[other]), seed, recipient.crop_id, "other", other)
            rows.append({**base, "condition": "unseen_other_class", "donor_id": donor, "donor_class": other})
    plan = pd.DataFrame(rows)
    plan["donor_parent"] = plan.donor_id.map(parent_of)
    plan["composite_id"] = plan.recipient_id + "|" + plan.condition + "|" + plan.donor_class
    if not plan.composite_id.is_unique:
        raise RuntimeError("composite identifiers are not unique")
    plan["seed"] = seed
    return plan


def build_transplant_cache(
    design: pd.DataFrame, cache: CropCache, output_dir: str, name: str
) -> tuple[dict, pd.DataFrame]:
    """Render every planned composite into a packed cache and return its plan."""
    plan = transplant_plan(design)
    backgrounds: dict[str, np.ndarray] = {}

    def background(crop_id: str) -> np.ndarray:
        if crop_id not in backgrounds:
            backgrounds[crop_id] = cell_removed_background(cache.array(crop_id))
        return backgrounds[crop_id]

    fractions = {}

    def items() -> Iterator[tuple[dict, np.ndarray]]:
        for recipient_id, group in plan.groupby("recipient_id", sort=True):
            recipient = cache.array(recipient_id)
            mask = foreground_mask(recipient)
            fractions[recipient_id] = float(1.0 - mask.mean())
            for row in group.itertuples(index=False):
                if row.condition == "original":
                    pixels = recipient
                else:
                    pixels = composite(recipient, background(row.donor_id), mask)
                yield {"crop_id": row.composite_id, "recipient_id": recipient_id, "condition": row.condition}, pixels

    manifest = write_array_cache(
        items(), output_dir, name, source={"kind": "background_transplants", "rows": len(plan)}
    )
    plan["background_fraction"] = plan.recipient_id.map(fractions)
    return manifest, plan
