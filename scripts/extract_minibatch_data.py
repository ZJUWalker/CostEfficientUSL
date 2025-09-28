import json
import csv
import os

def main():
    # 目标目录
    base_dir = "log/profile/meta-llama/llama3.2-1b"
    output_csv = "log/extract_minibatch_data_mbps_2000.csv"

    # micro batch size 值
    micro_batches = [1, 2, 4, 8, 16]

    # 存放结果
    rows = []

    for mb in micro_batches:
        filename = (
            f"sp_3_b_16_mb_{mb}_s_512_off_False_mbps_2000_sort_False_offms_False.json"
        )
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        row = {
            "micro_batch_size": mb,
            "max_mem_allocated_MB": data.get("max_mem_allocated_MB"),
            "batch_train_time_ms": data.get("batch_train_time_ms"),
            "client_idle_rate": data.get("client_idle_rate"),
            "server_idle_rate": data.get("server_idle_rate"),
        }
        rows.append(row)

    # 写 CSV
    os.makedirs("log", exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "micro_batch_size",
                "max_mem_allocated_MB",
                "batch_train_time_ms",
                "client_idle_rate",
                "server_idle_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已保存到 {output_csv}")


if __name__ == "__main__":
    main()
