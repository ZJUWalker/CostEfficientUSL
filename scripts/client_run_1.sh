#!/bin/bash
# client_run_1.sh
# 遍历 split_point(SP) ∈ {1..6}，带宽(mbps) ∈ {500,1000}
# 每次执行 client_run_1.py 时，同步触发 server_run_1.sh

for SP in 1 2 3 4 5 6; do
  for mbps in 500 1000; do
    echo ">>> Running SP=${SP}, mbps=${mbps}"

    # 启动服务器脚本（后台执行）
    bash scripts/server_run_1.sh $SP $mbps &

    # 启动客户端脚本（前台执行）
    python experiment/client_run.py -SP=$SP --mbps=$mbps --pmode=pdwc

    # 等待服务器进程结束
    wait
    echo ">>> Finished SP=${SP}, mbps=${mbps}"
    echo
  done
done
