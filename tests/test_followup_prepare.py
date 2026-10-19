import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")


def _module():
    path = Path(__file__).resolve().parents[1] / "analysis/study2/followup/prepare_followup.py"
    spec = importlib.util.spec_from_file_location("prepare_followup", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:  # the preparer imports torch-backed helpers
        pytest.skip(str(error))
    return module


def test_carrier_condition_reads_the_run_record():
    condition = _module().carrier_condition
    assert condition({"cache_name": "sipakmed", "color_augment": None}) == "standard"
    assert condition({"cache_name": "sipakmed"}) == "standard"
    assert condition({"cache_name": "sipakmed", "color_augment": "hed_strong_bc"}) == "colour"
    assert condition({"cache_name": "sipakmed_masked", "color_augment": None}) == "masked"
