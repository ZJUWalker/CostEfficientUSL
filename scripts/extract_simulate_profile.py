import os
import json
import csv

def extract_values(file_path, keys):
    """从 JSON 文件中提取指定键的值"""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    values = {}
    for key in keys:
        if key in data:
            try:
                values[key] = round(float(data[key]), 4)
            except (ValueError, TypeError):
                values[key] = None
        else:
            values[key] = None
    return values

def main():
    # 配置参数
    base_profile = "log/profile/meta-llama/llama3.2-1b"
    base_simulate = "log/simulate/meta-llama/llama3.2-1b"
    output_csv = "log/compare_profile_simulate_v1.csv"

    split_points = [1, 2, 3, 4, 5, 6]
    mbps_list = [500, 1000]
    offload_modes = [False, True]  # False: 无_oa_os, True: 有_oa_os

    # 准备输出
    results = []

    for sp in split_points:
        for mbps in mbps_list:
            for offload in offload_modes:
                offload_suffix = "_oa_os" if offload else ""
                file_name = f"sp_{sp}_b_8_mb_1_s_512_mbps_{mbps}_pipedream_wc{offload_suffix}.json"

                profile_path = os.path.join(base_profile, file_name)
                simulate_path = os.path.join(base_simulate, file_name)

                profile_keys = [
                    "client_max_mem_alloc_mb",
                    "server_max_mem_alloc_mb",
                    "batch_train_time_ms"
                ]
                simulate_keys = [
                    "client_peak_mem_alloc",
                    "server_peak_mem_alloc",
                    "batch_train_time"
                ]

                profile_data = extract_values(profile_path, profile_keys)
                simulate_data = extract_values(simulate_path, simulate_keys)

                if profile_data is None or simulate_data is None:
                    continue  # 有任意文件缺失则跳过

                # 计算差异百分比
                def compute_diff(a, b):
                    if a is None or b is None or a == 0:
                        return None
                    return abs((a - b) / a) * 100

                client_diff = compute_diff(
                    profile_data["client_max_mem_alloc_mb"],
                    simulate_data["client_peak_mem_alloc"]
                )
                server_diff = compute_diff(
                    profile_data["server_max_mem_alloc_mb"],
                    simulate_data["server_peak_mem_alloc"]
                )
                time_diff = compute_diff(
                    profile_data["batch_train_time_ms"],
                    simulate_data["batch_train_time"]
                )

                results.append({
                    "sp": sp,
                    "mbps": mbps,
                    "offload": offload,
                    "client_max_mem_alloc_mb": profile_data["client_max_mem_alloc_mb"],
                    "client_peak_mem_alloc": simulate_data["client_peak_mem_alloc"],
                    "server_max_mem_alloc_mb": profile_data["server_max_mem_alloc_mb"],
                    "server_peak_mem_alloc": simulate_data["server_peak_mem_alloc"],
                    "batch_train_time_ms": profile_data["batch_train_time_ms"],
                    "batch_train_time": simulate_data["batch_train_time"],
                    "client_diff(%)": round(client_diff, 5) if client_diff is not None else None,
                    "server_diff(%)": round(server_diff, 5) if server_diff is not None else None,
                    "time_diff(%)": round(time_diff, 5) if time_diff is not None else None,
                })

    # 写出结果 CSV
    os.makedirs("log", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sp", "mbps", "offload",
            "client_max_mem_alloc_mb", "client_peak_mem_alloc",
            "server_max_mem_alloc_mb", "server_peak_mem_alloc",
            "batch_train_time_ms", "batch_train_time",
            "client_diff(%)", "server_diff(%)", "time_diff(%)"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"结果已保存至 {output_csv} ，共 {len(results)} 条记录。")

if __name__ == "__main__":
    main()
