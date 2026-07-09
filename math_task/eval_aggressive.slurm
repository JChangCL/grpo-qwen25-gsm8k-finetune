#!/bin/bash
# ============================================================================
# Merge + MATH-500 eval for the aggressive MATH runs (Juno H100).
# For each run (order C -> D -> B -> A): merge the LoRA adapter into the base,
# then eval MATH-500 with the fixed is_correct. Prints RESULT lines to grep.
#   sbatch --export=ALL math_task/eval_aggressive.slurm
# ============================================================================
#SBATCH --job-name=eval-math-aggr
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

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONPATH="$PWD/math_task${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
PY=.venv-colocate/bin/python

# run_tag : run_dir_name : eval_max_tokens
RUNS=(
  "C:math-aggr-C-b01-lr2e5-500s-c1024:1024"
  "D:math-aggr-D-b01-lr1e5-500s-c1536:1536"
  "B:math-aggr-B-b01-lr1e5-500s-c1024:1024"
  "A:math-aggr-A-b02-lr1e5-400s-c1024:1024"
)

echo "######## base ref = 55.6% (fixed is_correct, job 264570) ########"
for R in "${RUNS[@]}"; do
  TAG="${R%%:*}"; REST="${R#*:}"; RN="${REST%%:*}"; MT="${REST##*:}"
  ADAPTER="outputs/qwen2.5-1.5b-math-grpo-${RN}"
  MERGED="outputs/merged-${RN}"
  echo "==================================================================="
  echo ">>> [$TAG] adapter=$ADAPTER  max_tokens=$MT"
  if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "!! [$TAG] no adapter_config.json in $ADAPTER — skipping"; continue
  fi
  # Merge on CPU, peft 0.15.2, WITHOUT distributed env vars (avoids DTensor error).
  env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
    "$PY" merge_lora.py --adapter_path "$ADAPTER" --output_dir "$MERGED" \
      --dtype bfloat16 --device_map cpu --overwrite \
    || { echo "!! [$TAG] merge failed"; continue; }
  # Eval MATH-500 (fixed parser via PYTHONPATH=math_task).
  "$PY" math_task/eval_math.py --model "$MERGED" --tag "aggr-$TAG" \
    --max_samples 500 --max_tokens "$MT" \
    || { echo "!! [$TAG] eval failed"; continue; }
  # Reclaim disk: the merged full model is ~3GB each.
  rm -rf "$MERGED"
  echo ">>> [$TAG] done"
done
echo "==================================================================="
echo ">>> ALL EVALS DONE — grep '^RESULT' in this log for the numbers"
