#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/runtime.sh"

ROLE=${ROLE:?Set ROLE=server or ROLE=client}
EXPERIMENT=${1:?usage: ROLE=server|client run.sh <experiment-directory> [--set key=value ...]}
shift
EXPERIMENT=$(resolve_experiment "$EXPERIMENT")

SERVER_WORLD_SIZE_OVERRIDE=""
index=1
while ((index <= $#)); do
  if [[ ${!index} != "--set" || index == $# ]]; then
    echo "Remote role overrides must use --set key=value pairs." >&2
    exit 2
  fi
  value_index=$((index + 1))
  value=${!value_index}
  if [[ $value == server.world_size=* ]]; then
    SERVER_WORLD_SIZE_OVERRIDE=${value#*=}
  fi
  index=$((index + 2))
done

PROFILE_ENABLED=$(bool_value "${SAVE_PROFILE:-1}")
CONTRACT="mode=run;experiment=$(basename -- "$EXPERIMENT");profile=$PROFILE_ENABLED"

case "$ROLE" in
  server)
    SERVER_WORLD_SIZE=${SERVER_WORLD_SIZE:-${SERVER_WORLD_SIZE_OVERRIDE:-1}}
    if ((SERVER_WORLD_SIZE < 1)); then
      echo "SERVER_WORLD_SIZE must be positive." >&2
      exit 2
    fi
    start_server_head "$SERVER_WORLD_SIZE"
    publish_remote_contract "$CONTRACT"
    if ! "$EXPERIMENTS_DIR/base/server.sh" "$EXPERIMENT/server.yaml" \
      "$@" \
      --set "server.world_size=$SERVER_WORLD_SIZE"; then
      ray stop --force >/dev/null 2>&1 || true
      exit 1
    fi
    echo "Server completed; run 'ray stop --force' after the client writes its artifacts."
    ;;
  client)
    start_client_worker
    verify_remote_contract "$CONTRACT"
    cleanup_client() {
      if [[ $(bool_value "${CLEANUP_CLIENT_RAY:-1}") == true ]]; then
        ray stop --force >/dev/null 2>&1 || true
      fi
    }
    trap cleanup_client EXIT INT TERM
    "$EXPERIMENTS_DIR/base/client.sh" "$EXPERIMENT/client.yaml" "$@"
    ;;
  *)
    echo "ROLE must be server or client, got: $ROLE" >&2
    exit 2
    ;;
esac
