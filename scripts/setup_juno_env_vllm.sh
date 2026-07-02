#!/bin/bash
set -euo pipefail

source scripts/setup_juno_env.sh

exec 8> .venv-vllm-setup.lock
flock 8

if ! python - <<'PY'
import importlib

vllm = importlib.import_module("vllm")
print("vllm:", getattr(vllm, "__version__", "unknown"))

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: F401
from vllm.distributed.utils import StatelessProcessGroup  # noqa: F401

print("vLLM training-side imports are available.")
PY
then
  python -m pip install "vllm==0.8.5.post1"
fi

python - <<'PY'
import vllm
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: F401
from vllm.distributed.utils import StatelessProcessGroup  # noqa: F401

print("vllm:", vllm.__version__)
print("vLLM training-side imports are available.")
PY
