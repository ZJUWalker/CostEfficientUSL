#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT=${1:?usage: run_grid.sh <experiment-config-directory>}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SPLIT_POINTS=(${SPLIT_POINTS:-3})
BATCH_SIZES=(${BATCH_SIZES:-8})
WORLD_SIZES=(${WORLD_SIZES:-1})
CLIENT_ACTIVATION_OFFLOAD=(${CLIENT_ACTIVATION_OFFLOAD:-0})
CLIENT_MODEL_OFFLOAD=(${CLIENT_MODEL_OFFLOAD:-0})
SERVER_ACTIVATION_OFFLOAD=(${SERVER_ACTIVATION_OFFLOAD:-0})

for split in "${SPLIT_POINTS[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do
    for world in "${WORLD_SIZES[@]}"; do
      for client_activation in "${CLIENT_ACTIVATION_OFFLOAD[@]}"; do
        for client_model in "${CLIENT_MODEL_OFFLOAD[@]}"; do
          for server_activation in "${SERVER_ACTIVATION_OFFLOAD[@]}"; do
            "$SCRIPT_DIR/run_single.sh" "$EXPERIMENT" \
              --set "split.split_point=$split" \
              --set "batch.batch_size=$batch" \
              --server-set "server.world_size=$world" \
              --server-set "server.activation_offload_mb=$server_activation" \
              --client-set "client.activation_offload_mb=$client_activation" \
              --client-set "client.model_state_offload_layers=$client_model"
          done
        done
      done
    done
  done
done
