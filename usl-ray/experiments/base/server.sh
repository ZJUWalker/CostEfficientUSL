#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

CONFIG=${1:?usage: server.sh <server.yaml> [--set key=value ...]}
shift
CONFIG=$(resolve_file "$CONFIG")
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address}"

PROFILE_ENABLED=$(bool_value "${SAVE_PROFILE:-1}")
cd "$PROJECT_DIR"
usl_cli server \
  --config "$CONFIG" \
  "$@" \
  --set "profile.enabled=$PROFILE_ENABLED"
