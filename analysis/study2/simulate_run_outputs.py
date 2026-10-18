#!/usr/bin/env python3
"""Fake run outputs with the trainer's exact schema, for dry-running the full analysis.

For every registered task this writes a run JSON, a per-crop prediction parquet
over the task's real split file, probe-crop features, and (for 2C cells)
transplant logits keyed by the committed composite IDs. Small effects are
planted (crop-level leakage, a context route for leak models, recoverability
falling with the reversal strength) so the dry run can check that each
registered model recovers the planted direction. The output is test scaffolding
and is never evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROOT = "/tmp/rchethan1_cytofair"
RUN_REPOSITORY = "/tmp/rchethan1_cytofair/run_repo"


def option(command: list[str], name: str, default=None):
    return command[command.index(name) + 1] if name in command else default


def options(command: list[str], name: str) -> list[str]:
    return [command[i + 1] for i, token in enumerate(command) if token == name]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.write_bytes(buffer.getvalue())


def logits_for(labels: np.ndarray, n_classes: int, p_correct: np.ndarray, rng) -> np.ndarray:
    """Logits whose argmax is correct with the given per-row probability."""
    logits = rng.normal(0, 1, (len(labels), n_classes))
    correct = rng.random(len(labels)) < p_correct
    wrong = (labels + rng.integers(1, n_classes, len(labels))) % n_classes
    winner = np.where(correct, labels, wrong)
    logits[np.arange(len(labels)), winner] += 6.0
    return logits


def parent_features(crop_ids, parents, strength: float, rng, dim: int = 32) -> np.ndarray:
    centres = {}
    for parent in sorted(set(parents)):
        digest = int.from_bytes(hashlib.sha256(parent.encode()).digest()[:4], "little")
        centres[parent] = np.random.default_rng(digest).normal(0, 1, dim)
    return np.stack([strength * centres[p] + rng.normal(0, 1, dim) for p in parents]).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--root", required=True, help="fake replacement for the training root")
    parser.add_argument("--tasks-out", required=True)
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()
    root, repository = Path(args.root).resolve(), Path(args.repository).resolve()
    tasks_out = Path(args.tasks_out)
    tasks_out.mkdir(parents=True, exist_ok=True)

    def local(path: str) -> Path:
        if path.startswith(RUN_REPOSITORY):
            return repository / path[len(RUN_REPOSITORY) + 1 :]
        if path.startswith(TRAIN_ROOT):
            return root / path[len(TRAIN_ROOT) + 1 :]
        return Path(path)

    provenance = pd.read_parquet(repository / "provenance/sipakmed_parent_map_p3.parquet")
    for task_file in args.tasks:
        rewritten = []
        for line in Path(task_file).read_text().splitlines():
            task = json.loads(line)
            command = task["command"]
            run_json = local(task["run_json"])
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(task["id"].encode()).digest()[:8], "little"))
            if task["experiment"] == "fingerprint":
                embedding = run_json.parent / "fingerprint_embeddings.npz"
                write_npz(embedding, crop_id=provenance.crop_id.to_numpy().astype(str),
                          features=parent_features(provenance.crop_id, provenance.parent_id, 1.0, rng))
                run_json.write_text(json.dumps({"task": "parent_fingerprint_encoder", "seed": 17}))
                rewritten.append({**task, "run_json": str(run_json)})
                continue
            expect = task["expect"]
            split_path = local(option(command, "--split"))
            split = pd.read_csv(split_path)
            split = split.loc[split.partition.isin(["train", "test"])]
            classes = sorted(split.class_name.unique())
            index = {name: i for i, name in enumerate(classes)}
            test = split.loc[split.partition == "test"].reset_index(drop=True)
            train = split.loc[split.partition == "train"]
            labels = test.class_name.map(index).to_numpy()
            protocol, lam = expect["protocol"], expect["adversary_lambda"]
            base = 0.95 if expect["dataset"] == "SIPaKMeD" else 0.80
            leak = protocol in {"P1", "P1-prime", "leak_high", "leak_low"}
            shrink = 1.0 if lam is None else max(0.0, 1 - lam / 1.5)
            p = base + (0.012 * shrink if leak else 0.0) + rng.normal(0, 0.004)
            logits = logits_for(labels, len(classes), np.full(len(labels), p), rng)
            predictions = logits.argmax(axis=1)
            output = local(option(command, "--output"))
            name = option(command, "--run-name")
            artifacts = {}
            prediction_path = output / "predictions" / f"{name}.parquet"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"crop_id": test.crop_id, "parent_id": test.parent_id, "label": labels,
                          "prediction": predictions,
                          **{f"logit_{i}": logits[:, i] for i in range(len(classes))}}).to_parquet(prediction_path)
            artifacts["predictions"] = {"path": str(prediction_path), "sha256": sha256(prediction_path)}
            feature_split = option(command, "--feature-split")
            if feature_split:
                probe = pd.read_csv(local(feature_split))
                strength = 1.2 * (1.0 if lam is None else max(0.05, 1 - lam / 3))
                feature_path = output / "features" / f"{name}.npz"
                write_npz(feature_path, crop_id=probe.crop_id.to_numpy().astype(str),
                          features=parent_features(probe.crop_id, probe.parent_id, strength, rng))
                artifacts["probe_features"] = {"path": str(feature_path), "sha256": sha256(feature_path)}
            for item in options(command, "--extra-eval"):
                seed = expect["seed"]
                plan = pd.read_csv(repository / f"splits/study2/sipakmed_2b/sipakmed_2b_plan_seed{seed}.csv")
                truth = plan.recipient_class.map(index).to_numpy()
                route = 0.01 if leak else 0.0
                p_correct = np.select(
                    [plan.condition.eq("sibling"), plan.condition.eq("unseen_other_class")],
                    [p, p - 0.05], default=p - route)
                extra_logits = logits_for(truth, len(classes), p_correct, rng)
                extra_path = output / "extra" / f"{name}.{item.split('=', 1)[0]}.npz"
                write_npz(extra_path, crop_id=plan.composite_id.to_numpy().astype(str),
                          logits=extra_logits.astype(np.float32))
                artifacts["extra:transplants"] = {"path": str(extra_path), "sha256": sha256(extra_path)}
            accuracy = float((predictions == labels).mean())
            confusion = pd.crosstab(pd.Series(labels), pd.Series(predictions)).reindex(
                index=range(len(classes)), columns=range(len(classes)), fill_value=0)
            record = {
                "study": "study2", "dataset": expect["dataset"], "task": "classification",
                "architecture": expect["architecture"], "protocol": protocol, "seed": expect["seed"],
                "accuracy": accuracy, "macro_f1": accuracy - 0.002, "balanced_accuracy": accuracy - 0.003,
                "nll": float(rng.uniform(0.1, 0.3)), "ece_15": float(rng.uniform(0.01, 0.03)),
                "confusion_matrix": confusion.to_numpy().tolist(), "class_names": classes,
                "train_crops": len(train), "test_crops": len(test),
                "train_parents": int(train.parent_id.nunique()), "test_parents": int(test.parent_id.nunique()),
                "test_crops_with_train_parent_fraction": float(test.parent_id.isin(set(train.parent_id)).mean()),
                "adversary_lambda": lam, "split_path": str(split_path), "split_file_sha256": expect["split_file_sha256"],
                "git_sha": "synthetic", "training_seconds": 1.0, "elapsed_seconds": 1.0, "gpu_visible_devices": "",
                "artifacts": artifacts, "history": [],
            }
            run_json.parent.mkdir(parents=True, exist_ok=True)
            run_json.write_text(json.dumps(record))
            rewritten.append({**task, "run_json": str(run_json)})
        (tasks_out / Path(task_file).name).write_text("".join(json.dumps(t) + "\n" for t in rewritten))
        print(f"{Path(task_file).name}: {len(rewritten)} synthetic runs")


if __name__ == "__main__":
    main()
