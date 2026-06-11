#!/bin/bash
#SBATCH --job-name=grpo-r32-200
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs .cache/huggingface .cache/wandb

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$PWD/.cache/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k-simulation}"

source scripts/setup_juno_env.sh

accelerate launch --config_file configs/accelerate_config.yaml train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-200step \
  --max_samples 2000 \
  --max_steps 200 \
  --learning_rate 2e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --max_completion_length 128 \
  --reward_weights 2.0 1.0 2.0 0.5 \
  --lora_r 32 \
  --lora_alpha 64 \
  --save_steps 10 \
  --fp16 true \
  --bf16 false \
  --use_lora true \
  --report_to wandb \
  --run_name utd-h100-qwen25-15b-gsm8k-grpo-4gen-r32-strong-200step

python eval_gsm8k.py \
  --model_name_or_path outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-200step \
  --max_samples 100

tar -czf "outputs/grpo-h100-4gen-r32-strong-200-results-${SLURM_JOB_ID}.tar.gz" \
  outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-200step logs .cache/wandb
