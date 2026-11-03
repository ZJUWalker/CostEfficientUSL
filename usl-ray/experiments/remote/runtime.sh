#!/usr/bin/env bash

REMOTE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$REMOTE_DIR/.." && pwd)
source "$EXPERIMENTS_DIR/base/common.sh"

SERVER_IP=${SERVER_IP:-10.124.0.24}
CLIENT_IP=${CLIENT_IP:-10.124.40.62}
RAY_PORT=${RAY_PORT:-26379}

start_server_head() {
  local slots=$1
  local physical_gpu_count
  physical_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  if ((slots < 1 || slots > physical_gpu_count)); then
    echo "Server requested $slots GPU(s), but this host exposes $physical_gpu_count." >&2
    return 2
  fi
  local visible
  visible=$(seq -s, 0 $((slots - 1)))
  export CUDA_VISIBLE_DEVICES="$visible"
  ray stop --force >/dev/null 2>&1 || true
  ray start --head \
    --node-ip-address="$SERVER_IP" \
    --port="$RAY_PORT" \
    --num-gpus="$slots" \
    --resources="{\"usl_server\": $slots}" \
    --disable-usage-stats
  export RAY_ADDRESS="$SERVER_IP:$RAY_PORT"
  echo "Ray head: $RAY_ADDRESS; server GPUs: $visible"
}

start_client_worker() {
  local client_gpu=${CLIENT_VISIBLE_DEVICE:-0}
  local physical_gpu_count
  physical_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  if [[ ! $client_gpu =~ ^[0-9]+$ ]] || ((client_gpu >= physical_gpu_count)); then
    echo "Invalid client GPU $client_gpu; this host exposes $physical_gpu_count GPU(s)." >&2
    return 2
  fi
  ray stop --force >/dev/null 2>&1 || true
  wait_for_tcp "$SERVER_IP" "$RAY_PORT" "${WAIT_SERVER_S:-600}"
  export CUDA_VISIBLE_DEVICES="$client_gpu"
  ray start \
    --address="$SERVER_IP:$RAY_PORT" \
    --node-ip-address="$CLIENT_IP" \
    --num-gpus=1 \
    --resources='{"usl_client": 1}' \
    --disable-usage-stats
  export RAY_ADDRESS="$SERVER_IP:$RAY_PORT"
  echo "Client worker: $CLIENT_IP; physical GPU: $client_gpu"
}

publish_remote_contract() {
  local signature=$1
  "$PYTHON_BIN" "$REMOTE_DIR/contract.py" publish "$signature" --address "$RAY_ADDRESS"
}

verify_remote_contract() {
  local signature=$1
  "$PYTHON_BIN" "$REMOTE_DIR/contract.py" verify "$signature" \
    --address "$RAY_ADDRESS" \
    --timeout-s "${WAIT_SERVER_S:-600}"
}
