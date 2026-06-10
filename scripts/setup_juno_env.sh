#!/bin/bash
set -euo pipefail

module purge
module load python/3.11.11
module load cuda/12.4

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio torchao timm || true
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-colab.txt

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
