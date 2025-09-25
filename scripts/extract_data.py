import json
import os
import pandas as pd

# 基础路径
base_path = 'log/profile/meta-llama/llama3.2-1b/'

# 输出文件路径
csv_output_path = 'log/extracted_data.csv'

# 所有可能的组合
sp_values = [1, 2, 3, 4]
mb_values = [1, 4]
mbps_values = [100, 500, 1000, 2000]

# 确保输出目录存在
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

# 存储所有提取的数据
all_data = []

# 遍历所有可能的文件组合
for sp in sp_values:
    for mb in mb_values:
        for mbps in mbps_values:
            # 构造文件名
            file_name = f"sp_{sp}_b_4_mb_{mb}_s_512_off_False_mbps_{mbps}.json"
            file_path = os.path.join(base_path, file_name)

            # 读取并提取数据
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    
                    # 提取需要的数据字段
                    extracted_data = {
                        "sp": sp,
                        "mb": mb,
                        "mbps": mbps,
                        "max_mem_allocated_MB": data.get("max_mem_allocated_MB", None),
                        "batch_train_time_ms": data.get("batch_train_time_ms", None),
                        "client_idle_rate": data.get("client_idle_rate", None),
                        "server_idle_rate": data.get("server_idle_rate", None),
                    }

                    # 将数据添加到所有数据列表
                    all_data.append(extracted_data)

                except json.JSONDecodeError:
                    print(f"文件格式错误: {file_path}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误: {e}")
            else:
                print(f"文件未找到: {file_path}")

# 如果有提取的数据，保存为 CSV 文件
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(csv_output_path, index=False)
    print(f"数据已成功保存为CSV文件: {csv_output_path}")
else:
    print("没有找到有效的数据文件")
