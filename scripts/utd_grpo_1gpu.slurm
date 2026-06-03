#!/bin/bash
#SBATCH --job-name=grpo-qwen15b
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs .cache/huggingface .cache/wandb

# Keep model/dataset caches in the project or scratch path instead of filling $HOME.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$PWD/.cache/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k-simulation}"

# Optional: load modules if UTD's cluster provides them.
# Run `module avail python`, `module avail cuda`, and edit these lines if needed.
module purge
module load python/3.11.11
module load cuda/12.4

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip uninstall -y torchao timm || true
python -m pip install -r requirements-colab.txt

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch not found in this environment; installing torch from PyPI."
  python -m pip install torch
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

accelerate launch --config_file configs/accelerate_config.yaml train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/qwen2.5-1.5b-gsm8k-grpo-hpc-8gen-200step \
  --max_samples 2000 \
  --max_steps 200 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_generations 8 \
  --max_completion_length 192 \
  --fp16 true \
  --bf16 false \
  --use_lora true \
  --report_to wandb \
  --run_name utd-hpc-qwen25-15b-gsm8k-grpo-8gen-200step

python eval_gsm8k.py \
  --model_name_or_path outputs/qwen2.5-1.5b-gsm8k-grpo-hpc-8gen-200step \
  --max_samples 100

tar -czf "outputs/grpo-hpc-results-${SLURM_JOB_ID}.tar.gz" outputs logs .cache/wandb
