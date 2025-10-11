SP_LIST=(1 2)
MBPS=$1
BS=$2
MODEL_NAME=$3
LORA=$4

# 遍历组合
for SP in "${SP_LIST[@]}"; do
    echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS"
    # 启动 server_run.sh，并传入参数（保证一一对应）
    bash scripts/server_run.sh $SP $MBPS $MODEL_NAME $LORA &
    SERVER_PID=$!

    # 启动 client
    python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc --model=$MODEL_NAME $LORA --step=3

    # 等待 server 执行完成
    wait $SERVER_PID
done
for SP in "${SP_LIST[@]}"; do
    echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS"
    # 启动 server_run.sh，并传入参数（保证一一对应）
    bash scripts/server_run.sh $SP $MBPS $MODEL_NAME $LORA &
    SERVER_PID=$!

    # 启动 client
    python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc --model=$MODEL_NAME -OA -OS $LORA --step=3

    # 等待 server 执行完成
    wait $SERVER_PID
done