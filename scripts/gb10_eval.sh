#!/usr/bin/env bash
# Same-protocol GSM8K eval on gb10b via vLLM chat.
#
#   MODE=grpo : merge a LoRA checkpoint -> serve merged model -> eval  (time-shares
#               the GPU with the base eval server: stops it, restores it after)
#   MODE=base : eval the base model against the already-running base vLLM server
#
# Usage:
#   ADAPTER_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-gb10-amd-8gen-r32-200step \
#   MAX_SAMPLES=500 bash scripts/gb10_eval.sh
#   MODE=base MAX_SAMPLES=500 bash scripts/gb10_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-grpo}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
IMAGE="${IMAGE:-grpo-gb10:latest}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.05.post1-py3}"
BASE_CONTAINER="${BASE_CONTAINER:-qwen15b-base-vllm}"
mkdir -p logs outputs

if [ "$MODE" = "base" ]; then
  BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
  MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
  echo ">>> base eval ($MAX_SAMPLES, chat) against $BASE_URL"
  docker run --rm --network host -v "$PWD":/workspace -w /workspace "$IMAGE" \
    python eval_gsm8k_vllm_api.py --base_url "$BASE_URL" --model "$MODEL_NAME" \
      --endpoint chat --max_samples "$MAX_SAMPLES" \
      --output_jsonl "logs/eval-base-chat-${MAX_SAMPLES}.jsonl" | tee "logs/eval-base-${MAX_SAMPLES}.out"
  exit 0
fi

# ---- MODE=grpo ----
ADAPTER_PATH="${ADAPTER_PATH:?set ADAPTER_PATH=outputs/<run> (LoRA checkpoint dir)}"
MERGED_DIR="${MERGED_DIR:-${ADAPTER_PATH%/}-merged}"
MODEL_NAME="${MODEL_NAME:-grpo-merged}"
PORT="${PORT:-8000}"
SERVE_UTIL="${SERVE_UTIL:-0.5}"

echo ">>> [1/4] merge LoRA -> $MERGED_DIR"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v "$PWD":/workspace -v "$HOME/hf-cache":/root/.cache/huggingface -w /workspace \
  "$IMAGE" python merge_lora.py --adapter_path "$ADAPTER_PATH" --output_dir "$MERGED_DIR" --dtype bf16 --overwrite

echo ">>> [2/4] free GPU: stop base eval server ($BASE_CONTAINER)"
docker stop "$BASE_CONTAINER" >/dev/null 2>&1 || echo "(base not running)"
restore_base() { echo ">>> restarting base eval server"; docker start "$BASE_CONTAINER" >/dev/null 2>&1 || true; }
trap 'docker stop grpo-eval-serve >/dev/null 2>&1 || true; restore_base' EXIT

echo ">>> [3/4] serve merged model on :$PORT"
docker run -d --rm --name grpo-eval-serve --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v "$PWD":/workspace -v "$HOME/hf-cache":/root/.cache/huggingface -w /workspace \
  "$VLLM_IMAGE" vllm serve "/workspace/$MERGED_DIR" --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" --max-model-len 2048 --gpu-memory-utilization "$SERVE_UTIL" --enforce-eager >/dev/null
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "server up"; break; }
  sleep 5
done

echo ">>> [4/4] eval $MAX_SAMPLES (chat) on merged GRPO model"
docker run --rm --network host -v "$PWD":/workspace -w /workspace "$IMAGE" \
  python eval_gsm8k_vllm_api.py --base_url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" \
    --endpoint chat --max_samples "$MAX_SAMPLES" \
    --output_jsonl "logs/eval-${MODEL_NAME}-chat-${MAX_SAMPLES}.jsonl" | tee "logs/eval-${MODEL_NAME}-${MAX_SAMPLES}.out"
echo ">>> done (base server will be restored on exit)"
