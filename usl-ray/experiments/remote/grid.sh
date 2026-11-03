#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/runtime.sh"

ROLE=${ROLE:?Set ROLE=server or ROLE=client}
EXPERIMENT=${1:-"$EXPERIMENTS_DIR/configs/smoke-qwen3-0.6b"}
EXPERIMENT=$(resolve_experiment "$EXPERIMENT")

read -r -a SPLITS <<< "${SPLIT_POINTS:-4}"
read -r -a BATCHES <<< "${BATCH_SIZES:-8}"
read -r -a WORLDS <<< "${WORLD_SIZES:-1}"
read -r -a CLIENT_ACTIVATIONS <<< "${CLIENT_ACTIVATION_OFFLOAD:-0}"
read -r -a CLIENT_MODELS <<< "${CLIENT_MODEL_OFFLOAD:-0}"
read -r -a SERVER_ACTIVATIONS <<< "${SERVER_ACTIVATION_OFFLOAD:-0}"
GRID_RUN_PREFIX=${GRID_RUN_PREFIX:-remote-grid}
PROFILE_ENABLED=$(bool_value "${SAVE_PROFILE:-1}")
CONTRACT="mode=grid;experiment=$(basename -- "$EXPERIMENT");profile=$PROFILE_ENABLED;splits=${SPLITS[*]};batches=${BATCHES[*]};worlds=${WORLDS[*]};client_activation=${CLIENT_ACTIVATIONS[*]};client_model=${CLIENT_MODELS[*]};server_activation=${SERVER_ACTIVATIONS[*]};prefix=$GRID_RUN_PREFIX"

case "$ROLE" in
  server)
    MAX_WORLD_SIZE=0
    for world in "${WORLDS[@]}"; do
      if ((world < 1)); then
        echo "WORLD_SIZES must contain positive integers." >&2
        exit 2
      fi
      if ((world > MAX_WORLD_SIZE)); then
        MAX_WORLD_SIZE=$world
      fi
    done
    start_server_head "$MAX_WORLD_SIZE"
    publish_remote_contract "$CONTRACT"
    trap 'ray stop --force >/dev/null 2>&1 || true' ERR INT TERM
    ;;
  client)
    start_client_worker
    verify_remote_contract "$CONTRACT"
    trap 'ray stop --force >/dev/null 2>&1 || true' EXIT INT TERM
    ;;
  *)
    echo "ROLE must be server or client, got: $ROLE" >&2
    exit 2
    ;;
esac

for split in "${SPLITS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for world in "${WORLDS[@]}"; do
      for client_activation in "${CLIENT_ACTIVATIONS[@]}"; do
        for client_model in "${CLIENT_MODELS[@]}"; do
          for server_activation in "${SERVER_ACTIVATIONS[@]}"; do
            RUN_NAME="${GRID_RUN_PREFIX}-sp${split}-b${batch}-ws${world}-coa${client_activation}-cos${client_model}-soa${server_activation}"
            COMMON_ARGS=(
              --set "run.name=$RUN_NAME"
              --set run.mode=remote
              --set "split.split_point=$split"
              --set "batch.batch_size=$batch"
            )
            if [[ $ROLE == server ]]; then
              "$EXPERIMENTS_DIR/base/server.sh" "$EXPERIMENT/server.yaml" \
                "${COMMON_ARGS[@]}" \
                --set "server.world_size=$world" \
                --set "server.activation_offload_mb=$server_activation"
            else
              "$EXPERIMENTS_DIR/base/client.sh" "$EXPERIMENT/client.yaml" \
                "${COMMON_ARGS[@]}" \
                --set "client.activation_offload_mb=$client_activation" \
                --set "client.model_state_offload_layers=$client_model"
            fi
          done
        done
      done
    done
  done
done

if [[ $ROLE == server ]]; then
  echo "Remote grid completed; run 'ray stop --force' after the client exits."
fi
