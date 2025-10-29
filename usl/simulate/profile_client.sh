#!/bin/bash

MBPS=230
MODEL_NAME=qwen/qwen3-8b #qwen/qwen3-1.7b | meta-llama/llama3.2-1b
LORA="--lora"
MAX_SP=18 # 模型的层数//2
BS=8
SP_LIST=(2 3)
SAVE_DIR='log/profile/sim_profile'

run_exp() {
    local SP=$1
    local CLIENT_OFFLOAD_ARG=$2
    local DESC=$3

    echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS , $DESC"

    python experiment/client_run.py -P=8888 -SP=$SP \
        --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 \
        --pmode=pipedream_wc --model=$MODEL_NAME \
        $LORA $CLIENT_OFFLOAD_ARG --step=4 --save_dir=$SAVE_DIR
    sleep 3
}

for SP in "${SP_LIST[@]}"; do
    run_exp $SP "" "no offload" # base
    run_exp $SP "-OAM=$BS" "with activation offload"
    run_exp $SP "-OSSP=$SP" "with model state offload"
done
python experiment/client_run.py -P=8888 -SP=$((SP_LIST[0])) \
    --mbps=$MBPS --batch_size=$((BS + 1)) --micro_batch_size=1 \
    --pmode=pipedream_wc --model=$MODEL_NAME \
    --step=4 --save_dir=$SAVE_DIR -OA $LORA
echo ">>> Done!"
