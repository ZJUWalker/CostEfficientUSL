#!/bin/bash

SP=$1
MBPS=$2
MODEL_NAME=$3
LORA=$4

echo ">>> [SERVER] Running with SP=$SP, MBPS=$MBPS, MODEL_NAME=$MODEL_NAME"
python experiment/server_run.py -SP=$SP --mbps=$MBPS --model=$MODEL_NAME --pmode=pipedream_wc $LORA