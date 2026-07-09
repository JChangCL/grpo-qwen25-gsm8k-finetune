#!/bin/bash
# ============================================================================
# CONFIRMATION run for the winning MATH recipe C (β0.01/lr2e-5/500s/c1024),
# with a DIFFERENT SEED (123 vs the original 42) — checks the +2.0pt MATH-500
# gain (57.6% vs base 55.6%) isn't 500-sample noise (+2pt = 10 problems).
# Self-contained: train (colocate) -> merge LoRA (cpu) -> eval MATH-500 -> RESULT.
#
# Submit ONLINE (key now valid):
#   source ~/.wandb_key && sbatch --export=ALL math_task/sbatch_math_C_confirm_seed123.sh
# ============================================================================
#SBATCH --job-name=grpo-math-C-confirm
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

export WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k-simulation}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONPATH="$PWD/math_task${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
: "${WANDB_API_KEY:?export WANDB_API_KEY (source ~/.wandb_key) before sbatch for an ONLINE run.}"
PY=.venv-colocate/bin/python

# --- Winning recipe C, seed 123 ---------------------------------------------
DATASET="nlile/hendrycks-MATH-benchmark"
SEED=123
MAX_STEPS=500; LR=2e-5; BETA=0.01
NUM_GEN=8; MAX_COMPLETION=1024; PER_DEV_BS=16; GRAD_ACCUM=1; VLLM_UTIL=0.35
REWARD_WEIGHTS="2.0 0.5"; MAX_SAMPLES=4000
RUN_NAME="math-aggr-C-confirm-seed123"
OUTPUT_DIR="outputs/qwen2.5-1.5b-math-grpo-${RUN_NAME}"
MERGED="outputs/merged-${RUN_NAME}"

# --- TRAIN (colocate: needs triton headers + single-process dist env) -------
PYINC=$(find /opt/ohpc/pub/apps -name Python.h -path '*3.11*' -print -quit 2>/dev/null | xargs -r dirname || true)
export CPATH="${PYINC}:${CPATH:-}"; export C_INCLUDE_PATH="${PYINC}:${C_INCLUDE_PATH:-}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))
echo ">>> $(hostname): TRAIN C-confirm seed$SEED (b${BETA}/lr${LR}/${MAX_STEPS}s/c${MAX_COMPLETION})"
"$PY" math_task/train_grpo_math.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name "$DATASET" --output_dir "$OUTPUT_DIR" --seed "$SEED" \
  --max_samples "$MAX_SAMPLES" --max_steps "$MAX_STEPS" \
  --learning_rate "$LR" --beta "$BETA" \
  --per_device_train_batch_size "$PER_DEV_BS" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_generations "$NUM_GEN" --max_completion_length "$MAX_COMPLETION" \
  --reward_weights $REWARD_WEIGHTS \
  --lora_r 32 --lora_alpha 64 --vllm_gpu_memory_utilization "$VLLM_UTIL" \
  --save_steps 20 --report_to wandb --run_name "$RUN_NAME" \
  || { echo "!! train failed"; exit 1; }

# --- MERGE + EVAL (dist env vars UNSET: peft DTensor + clean vLLM standalone)
echo ">>> MERGE + EVAL C-confirm"
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
  "$PY" merge_lora.py --adapter_path "$OUTPUT_DIR" --output_dir "$MERGED" \
    --dtype bfloat16 --device_map cpu --overwrite || { echo "!! merge failed"; exit 1; }
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
  "$PY" math_task/eval_math.py --model "$MERGED" --tag "aggr-C-confirm-seed123" \
    --max_samples 500 --max_tokens 1024 || { echo "!! eval failed"; exit 1; }
rm -rf "$MERGED"
echo ">>> C-confirm done — grep '^RESULT' for MATH-500 (base 55.6%, C-seed42 was 57.6%)"
