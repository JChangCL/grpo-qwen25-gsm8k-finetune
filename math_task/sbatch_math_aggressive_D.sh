#!/bin/bash
# ============================================================================
# Aggressive MATH GRPO — Run MATH-D  (Juno H100, vLLM colocate)
#   beta=0.01  lr=1e-5  steps=500  max_completion_length=1536
# Hypothesis: same recipe as MATH-B but LONGER completions (1024->1536) to test
# whether truncation was starving the reward on hard MATH problems. If B shows
# a high clipped/truncation ratio, D should recover more correct rollouts.
#
# OOM MITIGATION (completion 1536 is the memory-heaviest run):
#   * per_device_train_batch_size 16 -> 8  and grad_accum 1 -> 2
#     (same effective optimizer batch = 16 completions = 2 prompt-groups,
#      but half the peak activation memory in the policy fwd/bwd).
#   * vllm_gpu_memory_utilization 0.35 -> 0.45  (longer gen => bigger KV cache).
#   If it still OOMs: drop VLLM_UTIL to 0.40 or PER_DEV_BS to 8/GRAD_ACCUM 4,
#   or lower MAX_COMPLETION to 1280.  max_prompt_length stays TRL-default 512,
#   so vLLM max_model_len = 512+1536 = 2048 (fits comfortably).
#
# Submit (W&B key kept out of argv):
#   printf '%s\n' "$WANDB_API_KEY" | ssh juno 'IFS= read -r K; \
#     WANDB_API_KEY=$K sbatch --export=ALL math_task/sbatch_math_aggressive_D.sh'
# ============================================================================
#SBATCH --job-name=grpo-math-D
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
# Unique port per job so co-scheduled jobs on one node do not collide (EADDRINUSE on 29500).
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))
echo ">>> triton python headers: ${PYINC:-NOT FOUND}"

# --- Run MATH-D hyperparameters (aggressive matrix, long-completion) --------
DATASET="nlile/hendrycks-MATH-benchmark"
MAX_STEPS=500
LR=1e-5
BETA=0.01
NUM_GEN=8
MAX_COMPLETION=1536
PER_DEV_BS=8                 # OOM mitigation: half of A/B/C's 16 ...
GRAD_ACCUM=2                 # ... x2 accum => same effective optimizer batch
VLLM_UTIL=0.45               # longer gen needs a bigger KV cache
REWARD_WEIGHTS="2.0 0.5"     # [correctness, boxed_format]  (fixed v2 reward)
MAX_SAMPLES=4000
RUN_NAME="math-aggr-D-b01-lr1e5-500s-c1536"
OUTPUT_DIR="outputs/qwen2.5-1.5b-math-grpo-${RUN_NAME}"

echo ">>> $(hostname): MATH-D on $DATASET (b${BETA}/lr${LR}/${MAX_STEPS}step/c${MAX_COMPLETION}) [bs${PER_DEV_BS}x${GRAD_ACCUM}, vllm${VLLM_UTIL}]"
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

echo ">>> MATH-D done; checkpoints in $OUTPUT_DIR"
echo ">>> next: merge_lora then math_task/eval_math.py (see AGGRESSIVE_MATH_PLAN.md)"
