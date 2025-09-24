#!/usr/bin/env python3
import sys
import subprocess
import os
import signal
import argparse


def free_port(port):
    try:
        # 用 lsof 查找占用端口的进程
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().splitlines()
        if not pids:
            print(f"端口 {port} 没有被占用。")
            return
        for pid in pids:
            print(f"杀掉占用端口 {port} 的进程 PID={pid}")
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception as e:
        print(f"释放端口 {port} 出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="释放端口占用")
    parser.add_argument("--port", type=int, help="端口号", default=8000)
    args = parser.parse_args()
    free_port(args.port)
