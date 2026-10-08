#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: run_server.sh <server.yaml> [--set key=value ...]}
shift
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address}"
usl-ray server --config "$CONFIG" "$@"
