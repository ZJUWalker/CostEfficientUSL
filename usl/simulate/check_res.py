#!/usr/bin/env python3
import argparse
import json
import os
import subprocess

KEYS = [
    'client_max_mem_alloc_mb',
    'server_max_mem_alloc_mb',
    'batch_train_time_ms',
    'client_idle_rate',
    'server_idle_rate',
]


def run_experiment(model_name, sp, bs, mbps, coam, soam, ossp, lora):
    """调用 Bash 脚本运行一次实验"""
    cmd = ["bash", "usl/simulate/check.sh", model_name, str(sp), str(bs), str(mbps), str(coam), str(soam), str(ossp), lora or ""]  # 你的脚本名

    print(">>> Running command:\n", " ".join(cmd), "\n")

    # process = subprocess.Popen(
    #     cmd,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     text=True,
    # )
    subprocess.run(cmd)

    # 实时打印输出
    # for line in process.stdout:
    #     print(line, end="")

    # process.wait()
    # print(f"\n>>> Script exited with code {process.returncode}")
    # return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Python launcher for run_experiment.sh")

    parser.add_argument("--model", default="qwen/qwen3-1.7b", help="模型名称，如 llama3-8b")
    parser.add_argument("--sp", type=int, default=4, help="split point")
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--mbps", type=int, default=230, help="网络带宽 (Mbps)")
    parser.add_argument("--coam", type=int, default=0, help="client offload activation memory")
    parser.add_argument("--soam", type=int, default=0, help="server offload activation memory")
    parser.add_argument("--ossp", type=int, default=3, help="offload model state split point")
    parser.add_argument("--lora", action="store_true", help="LORA 参数字符串（可选）")
    parser.add_argument("--force", action="store_true", help="强制重新运行实验")

    args = parser.parse_args()
    offload_str = f"_coa_{args.coam}" if args.coam > 0 else ""
    offload_str += f'_cos_{args.ossp}' if args.ossp > 0 else ""
    offload_str += f'_soa_{args.soam}' if args.soam > 0 else ""
    file_name = f"sp_{args.sp}_b_{args.bs}_mb_1_s_512_mbps_{args.mbps}_pipedream_wc{'_lora' if args.lora else '' }{offload_str}.json"
    file_path = os.path.join("log/profile", args.model, file_name)
    if not os.path.exists(file_path) or args.force:
        run_experiment(
            model_name=args.model,
            sp=args.sp,
            bs=args.bs,
            mbps=args.mbps,
            coam=args.coam,
            soam=args.soam,
            ossp=args.ossp,
            lora="--lora" if args.lora else "",
        )
    try:
        print(f"Profile file path: {file_path}")
        with open(file_path, "r") as f:
            res = json.load(f)
        for key in KEYS:
            print(f"{key}: {round(res[key],2)}")

    except FileNotFoundError:
        print("Profile file not found.")


if __name__ == "__main__":
    main()
