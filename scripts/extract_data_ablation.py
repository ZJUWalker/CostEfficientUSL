import json
import os
import pandas as pd

# 基础路径
base_path = 'log/profile/meta-llama/llama3.2-1b/'

# 输出文件路径
csv_output_path = 'log/extracted_data.csv'
split_points = [1, 2, 4]
mbps_values = [200, 300, 500, 1000]

batch_sizes = [8, 16]
all_data = []
# 确保输出目录存在
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
# 遍历所有文件
for sp in split_points:
    for batch_size in batch_sizes:
        for mbps in mbps_values:
            for offload in [False, True]:
                # 读取文件
                file_name = (
                    f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc.json"
                    if not offload
                    else f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc_oa_os.json"
                )
                file_path = os.path.join(base_path, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 解析数据
                # 提取需要的数据字段
                extracted_data = {
                    "split_point": sp,
                    "batch_size": data.get("batch_size", None),
                    "mbps": mbps,
                    "batch_train_time_ms": data.get("batch_train_time_ms", None),
                    "client_send_rate": data.get("client_send_rate", None),
                    "server_send_rate": data.get("server_send_rate", None),
                    "client_compute_rate": 100 - float(data.get("client_idle_rate", None)),
                    "server_compute_rate": 100 - float(data.get("server_idle_rate", None)),
                    "client_max_mem_alloc_mb": data.get("client_max_mem_alloc_mb", None),
                }
                print(extracted_data)
                # 写入csv文件
                # 将数据添加到所有数据列表
                all_data.append(extracted_data)

# 如果有提取的数据，保存为 CSV 文件
if all_data:
    df = pd.DataFrame(all_data)
    df = df.round(2)
    df.to_csv(csv_output_path, index=False)
    print(f"数据已成功保存为CSV文件: {csv_output_path}")
else:
    print("没有找到有效的数据文件")
