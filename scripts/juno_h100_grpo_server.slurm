#!/bin/bash
# AMD-split GRPO on Juno H100: this job TRAINS on the H100 and offloads rollout
# generation to a TRL vLLM server running on GB10B (server mode). Start the
# GB10B server first:  ssh gb10b 'bash ~/grpo-qwen25-gsm8k-finetune/scripts/gb10_trl_vllm_serve.sh'
#
# Submit with the W&B key in the environment (kept out of argv):
#   printf '%s\n' "$WANDB_API_KEY" | ssh juno 'IFS= read -r K; \
#     WANDB_API_KEY=$K sbatch --export=ALL scripts/juno_h100_grpo_server.slurm'
#SBATCH --job-name=grpo-h100-amd-split
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

export WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k-simulation}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported into the job (sbatch --export=ALL).}"

# --- GB10B TRL vLLM rollout server (the AMD split) ---
VLLM_HOST="${VLLM_HOST:-10.180.72.205}"   # gb10b
VLLM_PORT="${VLLM_PORT:-8000}"

RUN_NAME="${RUN_NAME:-juno-h100-amd-split-r32-8gen-150step}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-1.5b-gsm8k-grpo-${RUN_NAME}}"

echo ">>> preflight: is the GB10B TRL vLLM server reachable?"
python scripts/check_trl_vllm_server.py --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --expected-model Qwen/Qwen2.5-1.5B-Instruct

echo ">>> launch server-mode GRPO training on $(hostname)"
.venv-h100/bin/python train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir "$OUTPUT_DIR" \
  --max_samples 2000 --max_steps 150 \
  --learning_rate 1e-6 --beta 0.06 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --num_generations 8 --max_completion_length 128 \
  --reward_weights 2.0 1.0 2.0 0.5 \
  --lora_r 32 --lora_alpha 64 \
  --bf16 true --fp16 false --use_lora true --gradient_checkpointing true \
  --use_vllm true --vllm_mode server \
  --vllm_server_host "$VLLM_HOST" --vllm_server_port "$VLLM_PORT" \
  --save_steps 10 --save_total_limit 20 \
  --report_to wandb --run_name "$RUN_NAME"

echo ">>> training finished; checkpoints in $OUTPUT_DIR"
