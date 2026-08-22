import json

import pandas as pd

from cytofair.results import summarize_classification, verify_classification_matrix


def test_classification_summary_uses_five_seeds_and_one_bh_family(tmp_path):
    paths = []
    for protocol, accuracy in (("P1", 0.9), ("P1-prime", 0.85), ("P2", 0.8), ("P3", 0.75)):
        for seed in (17, 29, 43, 71, 101):
            path = tmp_path / f"{protocol}_{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "d",
                        "architecture": "resnet18",
                        "protocol": protocol,
                        "seed": seed,
                        "accuracy": accuracy + (seed % 7) / 1000,
                        "macro_f1": accuracy - 0.01,
                        "balanced_accuracy": accuracy - 0.02,
                    }
                )
            )
            paths.append(path)
    summary, primary, secondary, ordering, ranking = summarize_classification(
        paths, bootstrap_repetitions=20
    )
    assert len(summary) == 12
    assert summary.complete.all()
    assert len(primary) == 2
    assert primary.q_value.notna().all()
    assert primary.multiplicity_family.nunique() == 1
    assert secondary.effect_size.notna().all()
    assert ordering.ordering_passed.all()
    assert ranking.empty


def test_matrix_verifier_matches_semantic_split_hash_and_runtime(tmp_path):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    rows = pd.DataFrame(
        {
            "crop_id": ["a", "b"],
            "parent_id": ["x", "y"],
            "class_name": ["c", "c"],
            "partition": ["train", "test"],
        }
    )
    semantic_hash = __import__("hashlib").sha256(
        rows[["crop_id", "partition"]].to_csv(index=False).encode()
    ).hexdigest()
    paths = []
    for protocol in ("P1", "P1-prime"):
        split = rows.assign(protocol=protocol, seed=17)
        split.to_csv(split_root / f"d_{protocol.lower().replace('-', '_')}_seed17.csv", index=False)
        path = tmp_path / f"{protocol}.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": "D",
                    "task": "classification",
                    "architecture": "a",
                    "protocol": protocol,
                    "seed": 17,
                    "accuracy": 0.5,
                    "macro_f1": 0.5,
                    "balanced_accuracy": 0.5,
                    "train_crops": 1,
                    "test_crops": 1,
                    "split_sha256": semantic_hash,
                    "git_sha": "abc",
                    "gpu_visible_devices": "1",
                    "runtime": {"version": "fixed", "input_provenance_sha256": "source"},
                }
            )
        )
        paths.append(path)
    evidence = verify_classification_matrix(
        paths,
        split_root=split_root,
        dataset="D",
        architectures=["a"],
        protocols=["P1", "P1-prime"],
        seeds=[17],
        expected_git_sha="abc",
        expected_runtime={"version": "fixed"},
        expected_train_crops=1,
        expected_test_crops=1,
    )
    assert evidence["passed"]


def test_matrix_verifier_fails_on_wrong_environment(tmp_path):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    split = pd.DataFrame({"crop_id": ["a"], "partition": ["test"]})
    split.to_csv(split_root / "d_p1_seed17.csv", index=False)
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "D",
                "task": "classification",
                "architecture": "a",
                "protocol": "P1",
                "seed": 17,
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "balanced_accuracy": 0.5,
                "train_crops": 0,
                "test_crops": 1,
                "split_sha256": "wrong",
                "git_sha": "abc",
                "runtime": {"version": "old", "input_provenance_sha256": "source"},
            }
        )
    )
    evidence = verify_classification_matrix(
        [path],
        split_root=split_root,
        dataset="D",
        architectures=["a"],
        protocols=["P1"],
        seeds=[17],
        expected_git_sha="abc",
        expected_runtime={"version": "fixed"},
        expected_train_crops=0,
        expected_test_crops=1,
    )
    assert not evidence["passed"]
    assert any("runtime version" in error for error in evidence["errors"])
