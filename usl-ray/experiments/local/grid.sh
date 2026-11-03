#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXPERIMENT=${1:-"$EXPERIMENTS_DIR/configs/smoke-qwen3-0.6b"}

read -r -a SPLITS <<< "${SPLIT_POINTS:-4}"
read -r -a BATCHES <<< "${BATCH_SIZES:-8}"
read -r -a WORLDS <<< "${WORLD_SIZES:-1}"
read -r -a CLIENT_ACTIVATIONS <<< "${CLIENT_ACTIVATION_OFFLOAD:-0}"
read -r -a CLIENT_MODELS <<< "${CLIENT_MODEL_OFFLOAD:-0}"
read -r -a SERVER_ACTIVATIONS <<< "${SERVER_ACTIVATION_OFFLOAD:-0}"

for split in "${SPLITS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for world in "${WORLDS[@]}"; do
      for client_activation in "${CLIENT_ACTIVATIONS[@]}"; do
        for client_model in "${CLIENT_MODELS[@]}"; do
          for server_activation in "${SERVER_ACTIVATIONS[@]}"; do
            SERVER_WORLD_SIZE=$world "$SCRIPT_DIR/run.sh" "$EXPERIMENT" \
              --set "split.split_point=$split" \
              --set "batch.batch_size=$batch" \
              --server-set "server.activation_offload_mb=$server_activation" \
              --client-set "client.activation_offload_mb=$client_activation" \
              --client-set "client.model_state_offload_layers=$client_model"
          done
        done
      done
    done
  done
done
