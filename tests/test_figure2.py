import importlib.util
from pathlib import Path

import pandas as pd


def test_figure2_requires_and_states_five_seed_control():
    path = Path(__file__).parents[1] / "scripts" / "generate_figure2.py"
    spec = importlib.util.spec_from_file_location("generate_figure2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    summary = pd.DataFrame(
        {
            "image_mode": [
                "rgb",
                "grayscale",
                "stain_normalized",
                "background_only",
                "highpass",
                "rgb",
            ],
            "shuffled_labels": [False, False, False, False, False, True],
            "n": [5] * 6,
            "mean": [0.25, 0.13, 0.17, 0.08, 0.11, 0.002],
            "sd": [0.01] * 6,
            "complete": [True] * 6,
        }
    )
    svg = module.figure_svg(summary, {"chance": 1 / 424, "passed": True})
    assert "424-way" in svg
    assert "shuffled-label 95% CI covers chance: True" in svg
    assert "25.00 ± 1.00%" in svg
    assert 'y="171"' in svg
