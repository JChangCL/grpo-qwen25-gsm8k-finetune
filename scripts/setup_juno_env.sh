#!/bin/bash
set -euo pipefail

module purge
module load python/3.11.11
module load cuda/12.4

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

exec 9> .venv-setup.lock
flock 9

source .venv/bin/activate
python -m ensurepip --upgrade || true

if ! python - <<'PY'
import importlib
import sys

required = {
    "torch": "2.6.0",
    "trl": None,
    "transformers": None,
    "accelerate": None,
    "peft": None,
    "datasets": None,
    "wandb": None,
}

for module_name, version_prefix in required.items():
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "")
    if version_prefix and not version.startswith(version_prefix):
        raise SystemExit(f"{module_name} version {version} does not start with {version_prefix}")

print("Existing .venv dependencies look usable; skipping pip install.")
PY
then
  python -m pip install --upgrade pip
  python -m pip uninstall -y torch torchvision torchaudio torchao timm || true
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r requirements-colab.txt
fi

python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    sys.exit("CUDA is not available. Refusing to run training on CPU.")
PY
