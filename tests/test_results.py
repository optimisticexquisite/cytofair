import json

from cytofair.results import summarize_classification


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
    summary, primary, secondary, ordering = summarize_classification(
        paths, bootstrap_repetitions=20
    )
    assert len(summary) == 12
    assert summary.complete.all()
    assert len(primary) == 2
    assert primary.q_value.notna().all()
    assert primary.multiplicity_family.nunique() == 1
    assert secondary.effect_size.notna().all()
    assert ordering.ordering_passed.all()
