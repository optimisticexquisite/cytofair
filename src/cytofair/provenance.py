"""Canonical provenance schema and fail-closed validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("crop_id", "parent_id", "class_name", "method")
BBOX_COLUMNS = ("x", "y", "width", "height", "parent_width", "parent_height")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    observed: object
    expected: object | None = None
    detail: str = ""


@dataclass
class ProvenanceValidation:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(check) for check in self.checks)

    def raise_for_failure(self) -> None:
        failures = [check for check in self.checks if not check.passed]
        if failures:
            rendered = "; ".join(f"{c.name}: {c.detail or c.observed}" for c in failures)
            raise ValueError(f"Provenance validation failed: {rendered}")


def normalise_map(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic canonical map without weakening missing-data checks."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required provenance columns: {missing}")
    out = frame.copy()
    for col in ("crop_id", "parent_id", "class_name", "method"):
        out[col] = out[col].astype("string")
    numeric = set(BBOX_COLUMNS) | {"crop_width", "crop_height", "match_confidence"}
    for col in sorted(numeric & set(out.columns)):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("crop_id", kind="stable").reset_index(drop=True)


def branch_agreement(branch_a: pd.DataFrame, branch_b: pd.DataFrame) -> tuple[float, int]:
    """Parent-ID agreement on the exact crop-ID intersection."""
    a = normalise_map(branch_a)[["crop_id", "parent_id"]].rename(
        columns={"parent_id": "parent_a"}
    )
    b = normalise_map(branch_b)[["crop_id", "parent_id"]].rename(
        columns={"parent_id": "parent_b"}
    )
    both = a.merge(b, on="crop_id", validate="one_to_one")
    if both.empty:
        return float("nan"), 0
    return float((both.parent_a == both.parent_b).mean()), len(both)


def validate_provenance(
    frame: pd.DataFrame,
    *,
    expected_total: int | None = None,
    expected_parent_counts: Mapping[str, int] | None = None,
    branch_a: pd.DataFrame | None = None,
    branch_b: pd.DataFrame | None = None,
    min_branch_agreement: float = 0.99,
    require_bbox: bool = False,
) -> ProvenanceValidation:
    """Validate all supplied acceptance criteria and return every check."""
    out = normalise_map(frame)
    checks: list[Check] = []

    unique = bool(out.crop_id.is_unique)
    checks.append(Check("unique_crop_id", unique, int(out.crop_id.nunique()), len(out)))
    missing_parent = int(out.parent_id.isna().sum() + (out.parent_id == "").sum())
    checks.append(Check("parent_id_complete", missing_parent == 0, missing_parent, 0))

    if expected_total is not None:
        checks.append(Check("dataset_total", len(out) == expected_total, len(out), expected_total))
        count_sum = int(out.groupby("parent_id", dropna=False).size().sum())
        checks.append(Check("per_parent_count_sum", count_sum == expected_total, count_sum, expected_total))

    if expected_parent_counts is not None:
        actual = out.groupby("class_name", observed=True).parent_id.nunique().to_dict()
        for class_name, expected in expected_parent_counts.items():
            observed = int(actual.get(class_name, 0))
            checks.append(
                Check(
                    f"parent_count:{class_name}",
                    observed == expected,
                    observed,
                    expected,
                )
            )

    bbox_available = all(col in out.columns for col in BBOX_COLUMNS)
    if require_bbox or bbox_available:
        checks.append(Check("bbox_columns_present", bbox_available, bbox_available, True))
    if bbox_available:
        vals = out.loc[:, BBOX_COLUMNS]
        finite = np.isfinite(vals.to_numpy(dtype=float)).all(axis=1)
        positive = (out.width > 0) & (out.height > 0)
        contained = (
            (out.x >= 0)
            & (out.y >= 0)
            & (out.x + out.width <= out.parent_width)
            & (out.y + out.height <= out.parent_height)
        )
        valid = finite & positive & contained
        checks.append(
            Check(
                "bbox_within_parent",
                bool(valid.all()),
                int(valid.sum()),
                len(out),
                detail=f"{int((~valid).sum())} invalid bounding boxes",
            )
        )

    if branch_a is not None and branch_b is not None:
        agreement, common = branch_agreement(branch_a, branch_b)
        checks.append(Check("branch_overlap_nonempty", common > 0, common, ">0"))
        checks.append(
            Check(
                "branch_parent_agreement",
                bool(common > 0 and agreement >= min_branch_agreement),
                agreement,
                min_branch_agreement,
                detail=f"agreement={agreement:.6f} on n={common}" if common else "no common crops",
            )
        )

    return ProvenanceValidation(checks)


def read_map(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"Unsupported provenance format: {path}")


def write_map(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalise_map(frame).to_parquet(path, index=False)

