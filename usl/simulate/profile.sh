MBPS=$1
# BS=$2
MODEL_NAME=$2
LORA=$3
MAX_SP=$4
BS_LIST=($5)
SP_LIST=($6)
SAVE_DIR=$7
# 遍历组合
for SP in "${SP_LIST[@]}"; do
    for BS in "${BS_LIST[@]}"; do
        echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS"
        # 启动 server_run.sh，并传入参数（保证一一对应）
        python experiment/server_run.py -SP=`expr $MAX_SP - $SP` --mbps=$MBPS --batch_size=$BS --model=$MODEL_NAME --pmode=pipedream_wc $LORA &
        SERVER_PID=$!

        # 启动 client
        python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc --model=$MODEL_NAME $LORA --step=4 --save_dir=$SAVE_DIR

        # 等待 server 执行完成
        wait $SERVER_PID
    done
done
for SP in "${SP_LIST[@]}"; do
    for BS in "${BS_LIST[@]}"; do
        echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS ,with offload"
        # 启动 server_run.sh，并传入参数（保证一一对应）
        # bash scripts/server_run.sh `expr $MAX_SP - $SP` $MBPS $MODEL_NAME $LORA -OA &
        python experiment/server_run.py -SP=`expr $MAX_SP - $SP` --mbps=$MBPS --batch_size=$BS --model=$MODEL_NAME --pmode=pipedream_wc $LORA -OA &
        SERVER_PID=$!

        # 启动 client
        python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc --model=$MODEL_NAME -OA -OS $LORA --step=4 --save_dir=$SAVE_DIR

        # 等待 server 执行完成
        wait $SERVER_PID
    done
done