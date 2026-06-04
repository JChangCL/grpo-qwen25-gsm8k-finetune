#!/bin/bash
#SBATCH --job-name=eval-grpo
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load python/3.11.11
module load cuda/12.4

source .venv/bin/activate

MODEL_PATH="${MODEL_PATH:-outputs/qwen2.5-1.5b-gsm8k-grpo-hpc-8gen-200step/checkpoint-25}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

python eval_gsm8k.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_samples "$MAX_SAMPLES"
