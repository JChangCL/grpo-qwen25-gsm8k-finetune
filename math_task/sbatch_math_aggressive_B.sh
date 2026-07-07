#!/bin/bash
# ============================================================================
# Aggressive MATH GRPO — Run MATH-B  (Juno H100, vLLM colocate)
#   beta=0.01  lr=1e-5  steps=500  max_completion_length=1024
# Hypothesis: same headroom test as MATH-A but looser KL leash (beta 0.02->0.01)
# and more optimization (500 steps). Expect the LARGEST KL of the c1024 runs.
# This is the primary "does the policy move enough to beat base 55.6%?" run.
#
# Submit (W&B key kept out of argv):
#   printf '%s\n' "$WANDB_API_KEY" | ssh juno 'IFS= read -r K; \
#     WANDB_API_KEY=$K sbatch --export=ALL math_task/sbatch_math_aggressive_B.sh'
# ============================================================================
#SBATCH --job-name=grpo-math-B
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
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

# vLLM colocate env fixes (see EXPERIMENT_LOG §8): triton needs Python.h to JIT
# its CUDA ext, and vLLM's single-process executor reads RANK/WORLD_SIZE.
PYINC=$(find /opt/ohpc/pub/apps -name Python.h -path '*3.11*' -print -quit 2>/dev/null | xargs -r dirname || true)
export CPATH="${PYINC}:${CPATH:-}"
export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
echo ">>> triton python headers: ${PYINC:-NOT FOUND}"

# --- Run MATH-B hyperparameters (aggressive matrix) -------------------------
DATASET="hendrycks/competition_math"
MAX_STEPS=500
LR=1e-5
BETA=0.01
NUM_GEN=8
MAX_COMPLETION=1024
PER_DEV_BS=16
GRAD_ACCUM=1
VLLM_UTIL=0.35
REWARD_WEIGHTS="2.0 0.5"     # [correctness, boxed_format]  (fixed v2 reward)
MAX_SAMPLES=4000
RUN_NAME="math-aggr-B-b01-lr1e5-500s-c1024"
OUTPUT_DIR="outputs/qwen2.5-1.5b-math-grpo-${RUN_NAME}"

echo ">>> $(hostname): MATH-B on $DATASET (b${BETA}/lr${LR}/${MAX_STEPS}step/c${MAX_COMPLETION})"
.venv-colocate/bin/python math_task/train_grpo_math.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name "$DATASET" --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" --max_steps "$MAX_STEPS" \
  --learning_rate "$LR" --beta "$BETA" \
  --per_device_train_batch_size "$PER_DEV_BS" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_generations "$NUM_GEN" --max_completion_length "$MAX_COMPLETION" \
  --reward_weights $REWARD_WEIGHTS \
  --lora_r 32 --lora_alpha 64 --vllm_gpu_memory_utilization "$VLLM_UTIL" \
  --save_steps 20 --report_to wandb --run_name "$RUN_NAME"

echo ">>> MATH-B done; checkpoints in $OUTPUT_DIR"
echo ">>> next: merge_lora then math_task/eval_math.py (see AGGRESSIVE_MATH_PLAN.md)"
