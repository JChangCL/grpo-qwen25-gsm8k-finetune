#!/bin/bash
# AMD-recipe GRPO on Juno H100, vLLM COLOCATE (single-cluster: H100 trains and
# generates rollouts on the same GPU). Uses .venv-colocate (trl 0.17 + vllm
# 0.8.5, driver-550 compatible). GB10B is used only for the chat eval afterwards.
#
# Submit with the W&B key in the environment (kept out of argv):
#   printf '%s\n' "$WANDB_API_KEY" | ssh juno 'IFS= read -r K; \
#     WANDB_API_KEY=$K sbatch --export=ALL scripts/juno_h100_grpo_colocate.slurm'
#SBATCH --job-name=grpo-h100-colocate
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
export TOKENIZERS_PARALLELISM=false
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported into the job (sbatch --export=ALL).}"

# vLLM colocate needs (1) Python dev headers so triton can JIT-compile its CUDA
# utils (the bare-metal node lacks python3.11-devel), and (2) single-process
# torch.distributed env vars (no torchrun/accelerate launcher here).
PYINC=$(find /opt/ohpc/pub/apps -name Python.h -path '*3.11*' -print -quit 2>/dev/null | xargs -r dirname || true)
export CPATH="${PYINC}:${CPATH:-}"
export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
echo ">>> triton python headers: ${PYINC:-NOT FOUND}"

# Hyperparameters (env-overridable for the fine-tuning loop).
MAX_STEPS="${MAX_STEPS:-150}";      LR="${LR:-1e-6}";            BETA="${BETA:-0.06}"
NUM_GEN="${NUM_GEN:-8}";            MAX_COMPLETION="${MAX_COMPLETION:-256}"
LORA_R="${LORA_R:-32}";            LORA_ALPHA="${LORA_ALPHA:-64}"
PER_DEV_BS="${PER_DEV_BS:-16}";    REWARD_WEIGHTS="${REWARD_WEIGHTS:-2.0 1.0 2.0 0.5}"
VLLM_UTIL="${VLLM_UTIL:-0.40}";    MAX_SAMPLES="${MAX_SAMPLES:-2000}"
RUN_NAME="${RUN_NAME:-juno-h100-colocate-r${LORA_R}-${NUM_GEN}gen-${MAX_STEPS}s-c${MAX_COMPLETION}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-1.5b-gsm8k-grpo-${RUN_NAME}}"

echo ">>> $(hostname): colocate GRPO r${LORA_R}/${NUM_GEN}gen/${MAX_STEPS}step/lr${LR}/beta${BETA}/compl${MAX_COMPLETION}"
.venv-colocate/bin/python train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" --max_steps "$MAX_STEPS" \
  --learning_rate "$LR" --beta "$BETA" \
  --per_device_train_batch_size "$PER_DEV_BS" --gradient_accumulation_steps 1 \
  --num_generations "$NUM_GEN" --max_completion_length "$MAX_COMPLETION" \
  --reward_weights $REWARD_WEIGHTS \
  --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
  --bf16 true --fp16 false --use_lora true --gradient_checkpointing true \
  --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization "$VLLM_UTIL" \
  --save_steps 10 --save_total_limit 20 \
  --report_to wandb --run_name "$RUN_NAME"

echo ">>> training finished; checkpoints in $OUTPUT_DIR"
