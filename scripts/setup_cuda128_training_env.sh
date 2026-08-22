#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${1:-python}"
environment="${2:-.venv-cu128}"

uv venv --python "$python_bin" --system-site-packages "$environment"
uv pip install --python "$environment/bin/python" \
  numpy==2.5.2 \
  pandas==3.0.5 \
  scipy==1.18.1 \
  scikit-learn==1.9.0 \
  pillow==12.3.0 \
  opencv-python-headless==5.0.0.93 \
  pyarrow==25.0.1
uv pip install --python "$environment/bin/python" --no-deps -e .

"$environment/bin/python" - <<'PY'
import cv2
import numpy
import pandas
import PIL
import scipy
import sklearn
import torch
import torchvision

expected = {
    "numpy": (numpy.__version__, "2.5.2"),
    "pandas": (pandas.__version__, "3.0.5"),
    "scipy": (scipy.__version__, "1.18.1"),
    "scikit-learn": (sklearn.__version__, "1.9.0"),
    "Pillow": (PIL.__version__, "12.3.0"),
    "OpenCV": (cv2.__version__, "5.0.0"),
}
for name, (observed, wanted) in expected.items():
    if observed != wanted:
        raise RuntimeError(f"{name}={observed}; expected {wanted}")
if not torch.cuda.is_available() or torch.version.cuda != "12.8":
    raise RuntimeError(
        f"CUDA-12.8 Torch is required; torch={torch.__version__}, cuda={torch.version.cuda}"
    )
print(
    "validated",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"cuda={torch.version.cuda}",
    torch.cuda.get_device_name(0),
)
PY
