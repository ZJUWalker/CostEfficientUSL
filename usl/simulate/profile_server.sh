#!/bin/bash

MBPS=230
MODEL_NAME=qwen/qwen3-8b #qwen/qwen3-1.7b | meta-llama/llama3.2-1b
LORA="--lora"
MAX_SP=18 # 模型的层数//2
BS=8
SP_LIST=(2 3)
# SAVE_DIR=$7

run_exp() {
    local SP=$1
    local SERVER_OFFLOAD_ARG=$2
    local DESC=$3

    echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS , $DESC"

    python experiment/server_run.py -SP=$((MAX_SP - SP)) \
        --mbps=$MBPS --batch_size=$BS --model=$MODEL_NAME \
        --pmode=pipedream_wc $LORA $SERVER_OFFLOAD_ARG 
}

for SP in "${SP_LIST[@]}"; do
    run_exp $SP "" "no offload" # base
    run_exp $SP "-OAM=$BS" "with activation offload"
    run_exp $SP "" "with model state offload"
done
#额外加一个
python experiment/server_run.py -SP=$((MAX_SP - SP_LIST[0])) \
    --mbps=$MBPS --batch_size=$((BS + 1)) --model=$MODEL_NAME \
    --pmode=pipedream_wc $LORA -OAM=$((BS + 1))

