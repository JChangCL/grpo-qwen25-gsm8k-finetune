#!/usr/bin/env bash
# Run a TRL-compatible vLLM rollout server on GB10B, for use as the rollout
# backend of a server-mode GRPO training job on the Juno H100 (the AMD split:
# H100 trains, GB10B generates). Frees the GPU by stopping the base eval server;
# restart it with:  docker start qwen15b-base-vllm
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-2048}"
BASE_CONTAINER="${BASE_CONTAINER:-qwen15b-base-vllm}"
NAME="${NAME:-trl-vllm-rollout}"
IMAGE="${IMAGE:-grpo-gb10:latest}"

echo ">>> stop base eval server ($BASE_CONTAINER) to free the GB10 GPU"
docker stop "$BASE_CONTAINER" >/dev/null 2>&1 || echo "(base not running)"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo ">>> launch trl vllm-serve ($MODEL) on 0.0.0.0:$PORT (util=$UTIL)"
docker run -d --name "$NAME" --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host -v "$HOME/hf-cache":/root/.cache/huggingface \
  "$IMAGE" trl vllm-serve --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization "$UTIL" --enforce-eager --max-model-len "$MAX_LEN"

echo ">>> waiting for TRL vLLM endpoints to come up..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1 || \
     curl -sf "http://127.0.0.1:$PORT/get_tensor_parallel_size/" >/dev/null 2>&1; then
    echo ">>> TRL vLLM server up on :$PORT"; break
  fi
  sleep 5
done
echo ">>> logs: docker logs -f $NAME   | stop: docker rm -f $NAME && docker start $BASE_CONTAINER"
