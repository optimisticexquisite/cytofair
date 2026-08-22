import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_launchers_are_hard_capped_at_three(monkeypatch):
    output = "\n".join(f"{index}, 0, 0" for index in range(8))
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: output)
    probe = _load_script("launch_probe_matrix.py")
    classification = _load_script("launch_classification_matrix.py")
    assert probe.idle_gpus()[:3] == [0, 1, 2]
    assert classification.idle_gpus()[:3] == [0, 1, 2]
