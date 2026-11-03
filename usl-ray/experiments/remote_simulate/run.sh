#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$EXPERIMENTS_DIR/base/common.sh"

CONFIG_INPUT=${1:?usage: run.sh <client.yaml|experiment-directory> [cost-model.json] [--set key=value ...]}
shift
if [[ -d $CONFIG_INPUT ]]; then
  CONFIG_INPUT="$CONFIG_INPUT/client.yaml"
fi
if [[ ! -f $CONFIG_INPUT ]]; then
  echo "Client config does not exist: $CONFIG_INPUT" >&2
  exit 2
fi
CONFIG_INPUT=$(resolve_file "$CONFIG_INPUT")

COST_MODEL=${COST_MODEL:-"$PROJECT_DIR/runs/cost_model.json"}
if (($#)) && [[ $1 != --* ]]; then
  COST_MODEL=$1
  shift
fi
if [[ ! -f $COST_MODEL ]]; then
  echo "Cost model does not exist: $COST_MODEL" >&2
  exit 2
fi
COST_MODEL=$(resolve_file "$COST_MODEL")

OUTPUT=${SIMULATION_OUTPUT:-"$PROJECT_DIR/runs/remote_simulation.json"}
OUTPUT=$("$PYTHON_BIN" -c "from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())" "$OUTPUT")
cd "$PROJECT_DIR"
usl_cli simulate \
  --config "$CONFIG_INPUT" \
  --cost-model "$COST_MODEL" \
  --output "$OUTPUT" \
  "$@"

GANTT=$("$PYTHON_BIN" -c "from pathlib import Path; import sys; print(Path(sys.argv[1]).with_suffix('.png'))" "$OUTPUT")
echo "Simulation: $OUTPUT"
echo "Gantt:      $GANTT"
