#!/usr/bin/env bash
# AMD-aligned GRPO training on a GB10 box (e.g. gb10b), using vLLM *colocate*
# rollout inside the grpo-gb10 image. Time-shares the single GB10 GPU with the
# base eval vLLM container by stopping it during training and restarting after.
#
# Usage (on the GB10 host):
#   export WANDB_API_KEY=...            # required for W&B logging
#   bash scripts/gb10_train.sh          # E02 defaults (8gen / 200step / r32 strong)
#
# Override any knob via env, e.g.:
#   RUN_NAME=gb10-amd-8gen-r32-stable LR=1e-6 BETA=0.06 \
#   REWARD_WEIGHTS="2.0 0.8 1.2 0.25" bash scripts/gb10_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?export WANDB_API_KEY before running (see HPC_UTD.md)}"

IMAGE="${IMAGE:-grpo-gb10:latest}"
RUN_NAME="${RUN_NAME:-gb10-amd-8gen-r32-200step}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-1.5b-gsm8k-grpo-${RUN_NAME}}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# --- AMD-aligned hyperparameters (E02) ---
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
MAX_STEPS="${MAX_STEPS:-200}"
NUM_GEN="${NUM_GEN:-8}"
MAX_COMPLETION="${MAX_COMPLETION:-256}"
LR="${LR:-2e-6}"
BETA="${BETA:-0.04}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
PER_DEV_BS="${PER_DEV_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-2.0 1.0 2.0 0.5}"
VLLM_UTIL="${VLLM_UTIL:-0.45}"
SAVE_STEPS="${SAVE_STEPS:-10}"

# --- GPU time-sharing with the base eval server ---
BASE_CONTAINER="${BASE_CONTAINER:-qwen15b-base-vllm}"
STOP_BASE="${STOP_BASE:-1}"

mkdir -p logs outputs

restart_base() {
  if [ "$STOP_BASE" = "1" ]; then
    echo ">>> Restarting base eval server ($BASE_CONTAINER)..."
    docker start "$BASE_CONTAINER" >/dev/null 2>&1 || echo "!! could not restart $BASE_CONTAINER (start it manually)"
  fi
}
trap restart_base EXIT

if [ "$STOP_BASE" = "1" ]; then
  echo ">>> Stopping base eval server ($BASE_CONTAINER) to free the GB10 GPU..."
  docker stop "$BASE_CONTAINER" >/dev/null 2>&1 || echo "(base container not running, continuing)"
fi

echo ">>> Launching GRPO training run: $RUN_NAME"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -e WANDB_API_KEY -e WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k-simulation}" \
  -v "$PWD":/workspace -v "$HOME/hf-cache":/root/.cache/huggingface -w /workspace \
  "$IMAGE" python train_grpo.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" --beta "$BETA" \
    --per_device_train_batch_size "$PER_DEV_BS" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GEN" --max_completion_length "$MAX_COMPLETION" \
    --reward_weights $REWARD_WEIGHTS \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --bf16 true --fp16 false --use_lora true --gradient_checkpointing true \
    --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --save_steps "$SAVE_STEPS" --save_total_limit 20 \
    --report_to wandb --run_name "$RUN_NAME" 2>&1 | tee "logs/${RUN_NAME}.out"

echo ">>> Training finished. Checkpoints in $OUTPUT_DIR"
