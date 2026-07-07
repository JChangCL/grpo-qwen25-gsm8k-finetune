#!/bin/bash
# GRPO on MATH (Juno H100, vLLM colocate). Submit with the W&B key in env:
#   printf '%s\n' "$WANDB_API_KEY" | ssh juno 'IFS= read -r K; \
#     WANDB_API_KEY=$K sbatch --export=ALL math_task/juno_h100_math.slurm'
#SBATCH --job-name=grpo-math
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
export PYTHONPATH="$PWD/math_task${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported into the job (sbatch --export=ALL).}"

PYINC=$(find /opt/ohpc/pub/apps -name Python.h -path '*3.11*' -print -quit 2>/dev/null | xargs -r dirname || true)
export CPATH="${PYINC}:${CPATH:-}"
export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
# Unique port per job so co-scheduled jobs on one node do not collide (EADDRINUSE on 29500).
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))

DATASET="${DATASET:-nlile/hendrycks-MATH-benchmark}"
MAX_STEPS="${MAX_STEPS:-300}"; LR="${LR:-5e-6}"; BETA="${BETA:-0.03}"
NUM_GEN="${NUM_GEN:-8}"; MAX_COMPLETION="${MAX_COMPLETION:-1024}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-2.0 0.5}"; MAX_SAMPLES="${MAX_SAMPLES:-4000}"
RUN_NAME="${RUN_NAME:-juno-h100-math-r32-8gen-${MAX_STEPS}s-c${MAX_COMPLETION}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-1.5b-math-grpo-${RUN_NAME}}"

echo ">>> $(hostname): MATH GRPO on $DATASET (${MAX_STEPS}step/c${MAX_COMPLETION}/lr${LR}/beta${BETA})"
.venv-colocate/bin/python math_task/train_grpo_math.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name "$DATASET" --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" --max_steps "$MAX_STEPS" \
  --learning_rate "$LR" --beta "$BETA" \
  --num_generations "$NUM_GEN" --max_completion_length "$MAX_COMPLETION" \
  --reward_weights $REWARD_WEIGHTS \
  --lora_r 32 --lora_alpha 64 --vllm_gpu_memory_utilization 0.35 \
  --save_steps 20 --report_to wandb --run_name "$RUN_NAME"

echo ">>> MATH training done; checkpoints in $OUTPUT_DIR"
