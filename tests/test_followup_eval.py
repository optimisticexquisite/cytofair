import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cytofair.followup_eval import (
    apply_eraser,
    fit_logistic,
    within_class_eraser,
    within_class_recoverability,
)
from cytofair.gpu_augment import COLOR_AUGMENTATIONS, color_jitter


def _parents(n_classes=3, parents_per_class=6, crops=8, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    class_centres = rng.normal(0, 3, (n_classes, dim))
    rows, parents, classes = [], [], []
    for c in range(n_classes):
        for p in range(parents_per_class):
            centre = class_centres[c] + rng.normal(0, 1.5, dim)
            rows.append(centre + rng.normal(0, 0.5, (crops, dim)))
            parents += [c * parents_per_class + p] * crops
            classes += [c] * crops
    return (torch.as_tensor(np.vstack(rows), dtype=torch.float32), torch.as_tensor(parents),
            torch.as_tensor(classes))


def test_full_rank_eraser_removes_within_class_parent_covariance():
    x, parents, classes = _parents()
    n_parents, n_classes = int(parents.max()) + 1, int(classes.max()) + 1
    full = n_parents - n_classes
    mean, erasers, spectrum = within_class_eraser(x, parents, classes, (0, 4, full))
    assert erasers[0] is None and spectrum[0] >= spectrum[-1]
    erased = apply_eraser(x, mean, erasers[full]).double()
    indicator = torch.nn.functional.one_hot(parents, n_parents).double()
    class_share = indicator.sum(0) / torch.nn.functional.one_hot(classes, n_classes).double().sum(0)[
        torch.as_tensor([p // 6 for p in range(n_parents)])
    ]
    within = indicator - torch.nn.functional.one_hot(classes, n_classes).double()[:, torch.as_tensor(
        [p // 6 for p in range(n_parents)])] * class_share
    cross = (erased - erased.mean(0)).T @ (within - within.mean(0)) / len(x)
    original = (x.double() - x.double().mean(0)).T @ (within - within.mean(0)) / len(x)
    assert cross.abs().max() < 1e-6 * max(1.0, float(original.abs().max()))
    # Class means survive: the eraser only removes within-class parent directions.
    class_gap = erased[classes == 0].mean(0) - erased[classes == 1].mean(0)
    assert float(class_gap.norm()) > 1.0


def test_device_logistic_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.linear_model")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    x, _, classes = _parents(seed=1)
    predict = fit_logistic(x, classes, 3)
    scaler = preprocessing.StandardScaler().fit(x.numpy())
    reference = sklearn.LogisticRegression(C=1.0, max_iter=5000, tol=1e-8).fit(scaler.transform(x.numpy()),
                                                                              classes.numpy())
    ours = predict(x).softmax(1).numpy()
    theirs = reference.predict_proba(scaler.transform(x.numpy()))
    assert np.abs(ours - theirs).max() < 1e-3


def test_within_class_recoverability_on_separable_parents():
    x, parents, classes = _parents(crops=4, seed=2)
    crop_ids = [f"c{i}" for i in range(len(x))]
    probe = pd.DataFrame({
        "crop_id": crop_ids,
        "parent_id": [f"p{int(p)}" for p in parents],
        "class_name": [f"k{int(c)}" for c in classes],
        "probe_partition": ["fit", "fit", "eval", "eval"] * (len(x) // 4),
    })
    metrics = within_class_recoverability(x, probe, {c: i for i, c in enumerate(crop_ids)})
    assert metrics["within_class_accuracy"] > 0.9
    assert metrics["within_class_floor"] == pytest.approx(1 / 6)


def test_color_jitter_identity_and_determinism():
    batch = torch.rand(4, 3, 16, 16) * 0.9 + 0.05
    same = color_jitter(batch, torch.Generator().manual_seed(0), hed_sigma=0.0, brightness=0.0, contrast=0.0)
    assert torch.allclose(same, batch, atol=1e-5)
    params = COLOR_AUGMENTATIONS["hed_strong_bc"]
    first = color_jitter(batch, torch.Generator().manual_seed(3), **params)
    second = color_jitter(batch, torch.Generator().manual_seed(3), **params)
    assert torch.equal(first, second) and not torch.allclose(first, batch)
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
