#!/bin/bash
#SBATCH --job-name=grpo-gb10-smk
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
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

VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-10.180.72.205}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
EXPECTED_VLLM_MODEL="${EXPECTED_VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
REPORT_TO="${REPORT_TO:-none}"

echo "Checking GB10 vLLM server at http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1/models"
curl -fsS "http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1/models"
echo

python scripts/check_trl_vllm_server.py \
  --host "${VLLM_SERVER_HOST}" \
  --port "${VLLM_SERVER_PORT}" \
  --expected-model "${EXPECTED_VLLM_MODEL}"

source scripts/setup_juno_env_vllm.sh

accelerate launch --config_file configs/accelerate_config.yaml train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/qwen2.5-1.5b-gsm8k-grpo-h100-gb10-vllm-smoke \
  --max_samples 32 \
  --max_steps 5 \
  --learning_rate 2e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_generations 4 \
  --max_completion_length 96 \
  --reward_weights 2.0 1.0 2.0 0.5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --save_steps 5 \
  --save_total_limit 2 \
  --fp16 true \
  --bf16 false \
  --use_lora true \
  --use_vllm true \
  --vllm_server_host "${VLLM_SERVER_HOST}" \
  --vllm_server_port "${VLLM_SERVER_PORT}" \
  --report_to "${REPORT_TO}" \
  --run_name utd-h100-qwen25-15b-gsm8k-grpo-gb10-vllm-smoke-5step
