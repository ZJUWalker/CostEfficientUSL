#!/bin/bash

MBPS=$1
MODEL_NAME=$2
LORA=$3
MAX_SP=$4
BS=$5
SP_LIST=($6)
SAVE_DIR=$7

run_exp() {
    local SP=$1
    local CLIENT_OFFLOAD_ARG=$2
    local SERVER_OFFLOAD_ARG=$3
    local DESC=$4

    echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS , $DESC"

    python experiment/server_run.py -SP=$((MAX_SP - SP)) \
        --mbps=$MBPS --batch_size=$BS --model=$MODEL_NAME \
        --pmode=pipedream_wc $LORA $SERVER_OFFLOAD_ARG &
    SERVER_PID=$!

    python experiment/client_run.py -SP=$SP \
        --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 \
        --pmode=pipedream_wc --model=$MODEL_NAME \
        $LORA $CLIENT_OFFLOAD_ARG --step=4 --save_dir=$SAVE_DIR

    wait $SERVER_PID
}

for SP in "${SP_LIST[@]}"; do
    run_exp $SP "" "" "no offload" # base
    run_exp $SP "-OAM=$BS" "-OAM=$BS" "with activation offload"
    run_exp $SP "-OSSP=$SP" "" "with model state offload"
done
echo ">>> Done!"
