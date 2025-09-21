#!/usr/bin/env python3
import argparse
import os
import time
import csv
import tempfile
from pathlib import Path
import io
import math
import socket
import struct
import threading
from typing import Optional, Tuple

import torch

# --------------------------- helpers ---------------------------


def dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype.is_floating_point or dtype in (torch.bfloat16,):
        return torch.finfo(dtype).bits // 8
    else:
        return torch.iinfo(dtype).bits // 8


def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def parse_sizes(spec: str):
    """
    支持:
      --sizes "1e8,5e8"         （按元素数）
      --sizes "4x4096x4096,..."（按形状）
    返回: 每个 size 的元素总数
    """
    out = []
    for item in spec.split(","):
        item = item.strip().lower()
        if "x" in item:
            n = 1
            for p in item.split("x"):
                n *= int(float(p))
            out.append(n)
        else:
            out.append(int(float(item)))
    return out


def median(lst):
    s = sorted(lst)
    m = len(s) // 2
    return s[m] if len(s) % 2 == 1 else 0.5 * (s[m - 1] + s[m])


# --------------------------- SSD I/O （含序列化） ---------------------------


def flush_and_fsync(fobj):
    fobj.flush()
    os.fsync(fobj.fileno())


def torch_save_to_file(t_cpu, filepath: Path):
    with open(filepath, "wb") as f:
        torch.save(t_cpu, f)
        flush_and_fsync(f)


def torch_load_from_file(filepath: Path):
    with open(filepath, "rb") as f:
        t = torch.load(f, map_location="cpu")
    return t


def time_gpu_to_ssd(t_gpu: torch.Tensor, filepath: Path):
    """
    计时范围：GPU->CPU (DtoH) + 序列化 + 写盘
    返回：wall_time_sec
    """
    start = time.perf_counter()
    # DtoH：non_blocking 需配合 pinned memory 才能真正异步。但这边我们只计总时间。
    t_cpu = t_gpu.detach().to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    torch_save_to_file(t_cpu, filepath)
    end = time.perf_counter()
    return end - start


def time_ssd_to_gpu(filepath: Path, device: torch.device):
    """
    计时范围：读盘 + 反序列化到CPU + HtoD
    返回：(wall_time_sec, t_gpu)
    """
    start = time.perf_counter()
    t_cpu = torch_load_from_file(filepath)
    t_gpu = t_cpu.to(device, non_blocking=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start, t_gpu


# --------------------------- 纯 GPU<->CPU 拷贝 ---------------------------


def time_gpu_to_cpu_only(t_gpu: torch.Tensor, make_pinned: bool):
    """
    只测 DtoH。返回:
      wall_sec:  主机端计时（含同步等）
      cuda_sec:   设备端 CUDA event 计时（更接近纯 memcpy）
    """
    # 先准备一个目标 CPU 张量（可选 pinned）
    shape = t_gpu.shape
    dtype = t_gpu.dtype
    if make_pinned:
        dst = torch.empty(shape, dtype=dtype, pin_memory=True)
    else:
        dst = torch.empty(shape, dtype=dtype)

    # CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # wall-clock
    wall_start = time.perf_counter()

    # 将拷贝提交到当前 stream
    start_evt.record()
    dst.copy_(t_gpu, non_blocking=True)  # DtoH
    end_evt.record()

    # 等待完成
    end_evt.synchronize()
    wall_end = time.perf_counter()

    cuda_ms = start_evt.elapsed_time(end_evt)  # 毫秒
    return (wall_end - wall_start), (cuda_ms / 1000.0)  # 返回秒


def time_cpu_to_gpu_only(numel: int, dtype: torch.dtype, device: torch.device, make_pinned: bool):
    """
    只测 HtoD。返回 (wall_sec, cuda_sec)
    """
    # 源 CPU 张量
    if make_pinned:
        src = torch.empty(numel, dtype=dtype, pin_memory=True).normal_()
    else:
        src = torch.empty(numel, dtype=dtype).normal_()

    # 目标 GPU
    dst = torch.empty(numel, dtype=dtype, device=device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_evt.record()
    dst.copy_(src, non_blocking=True)  # HtoD
    end_evt.record()
    end_evt.synchronize()
    wall_end = time.perf_counter()

    cuda_ms = start_evt.elapsed_time(end_evt)
    return (wall_end - wall_start), (cuda_ms / 1000.0)


# --------------------------- 网络传输（Loopback/远端） ---------------------------
# 说明：为了便于在单机上复现，默认实现了 loopback 回环的 echo server。
# 也可将 --net-role=server 在一台机器运行，--net-role=client 在另一台机器运行实现真网测。

HEADER_STRUCT = struct.Struct("!Q")  # 8 字节无符号长度（网络字节序）


def tensor_serialize_cpu(t_cpu: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t_cpu, buf)
    return buf.getvalue()


def tensor_deserialize_cpu(b: bytes) -> torch.Tensor:
    buf = io.BytesIO(b)
    return torch.load(buf, map_location="cpu")


def _recvall(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(min(1 << 20, n - len(data)))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        data.extend(chunk)
    return bytes(data)


def _sendall_with_rate(sock: socket.socket, data: bytes, chunk_bytes: int, rate_mbps: Optional[float]):
    if rate_mbps is None or rate_mbps <= 0:
        sock.sendall(data)
        return
    # 简单令牌桶：按 chunk 发送并 sleep 控速
    bytes_per_sec = rate_mbps * 1024 * 1024 / 8.0  # Mbps -> MBps (Bytes/s)
    start = time.perf_counter()
    sent = 0
    for i in range(0, len(data), chunk_bytes):
        part = data[i : i + chunk_bytes]
        sock.sendall(part)
        sent += len(part)
        expected_elapsed = sent / bytes_per_sec
        now = time.perf_counter()
        if expected_elapsed > (now - start):
            time.sleep(expected_elapsed - (now - start))


class EchoServer(threading.Thread):
    def __init__(self, host: str, port: int, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.stop_event = stop_event
        self._srv_sock = None

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            self._srv_sock = srv
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            srv.settimeout(0.5)
            while not self.stop_event.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                with conn:
                    # 收到长度
                    hdr = _recvall(conn, HEADER_STRUCT.size)
                    (nbytes,) = HEADER_STRUCT.unpack(hdr)
                    payload = _recvall(conn, nbytes)
                    # 直接回显（echo）
                    conn.sendall(hdr)
                    conn.sendall(payload)


def time_network_roundtrip(
    t_gpu: torch.Tensor,
    device: torch.device,
    host: str,
    port: int,
    chunk_bytes: int,
    rate_limit_mbps: Optional[float],
    include_gpu_dtoh_htod: bool,
) -> Tuple[float, int]:
    """
    计时范围（客户端）：
      include_gpu_dtoh_htod=True 时：GPU->CPU + 序列化 + 发送 + 接收 + 反序列化 + CPU->GPU
      否则：仅 序列化 + 发送 + 接收 + 反序列化（纯 CPU 模式）
    返回：(往返秒, 负载字节)
    """
    if include_gpu_dtoh_htod:
        src_cpu = t_gpu.detach().to("cpu", non_blocking=True)
        torch.cuda.synchronize()
    else:
        src_cpu = torch.empty_like(t_gpu.detach().cpu()).normal_()

    payload = tensor_serialize_cpu(src_cpu)
    total_bytes = len(payload)

    # client connect
    start = time.perf_counter()
    with socket.create_connection((host, port), timeout=30) as sock:
        # 先发长度，再发数据（可限速）
        sock.sendall(HEADER_STRUCT.pack(total_bytes))
        _sendall_with_rate(sock, payload, chunk_bytes, rate_limit_mbps)
        # 收回显
        hdr = _recvall(sock, HEADER_STRUCT.size)
        (nbytes_back,) = HEADER_STRUCT.unpack(hdr)
        assert nbytes_back == total_bytes
        payload_back = _recvall(sock, nbytes_back)

    # 反序列化 & 可选 CPU->GPU
    dst_cpu = tensor_deserialize_cpu(payload_back)
    if include_gpu_dtoh_htod:
        _ = dst_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start, total_bytes


def run_net_once(
    n_elements: int,
    dtype: torch.dtype,
    device: torch.device,
    host: str,
    port: int,
    chunk_bytes: int,
    rate_limit_mbps: Optional[float],
    include_gpu_path: bool,
):
    t_gpu = torch.empty(n_elements, dtype=dtype, device=device).normal_()
    # 预热
    _ = time_network_roundtrip(t_gpu, device, host, port, chunk_bytes, rate_limit_mbps, include_gpu_path)
    # 正式
    rtt_sec, nbytes = time_network_roundtrip(t_gpu, device, host, port, chunk_bytes, rate_limit_mbps, include_gpu_path)
    # 推导单向吞吐（估计）
    one_way_sec = rtt_sec / 2.0 if rtt_sec > 0 else float("inf")
    gb = nbytes / (1024**3)
    return {
        "mode": f"net_{'gpu' if include_gpu_path else 'cpu'}",
        "elements": n_elements,
        "dtype": str(dtype).replace("torch.", ""),
        "tensor_bytes": n_elements * dtype_nbytes(dtype),
        "wire_bytes": nbytes,
        "rtt_sec": rtt_sec,
        "est_oneway_sec": one_way_sec,
        "est_oneway_GBps": gb / one_way_sec if one_way_sec and not math.isinf(one_way_sec) else float("inf"),
        "rate_limit_Mbps": rate_limit_mbps or 0.0,
        "chunk_bytes": chunk_bytes,
        "host": host,
        "port": port,
    }


# --------------------------- 单次任务 ---------------------------


def run_ssd_once(n_elements: int, dtype: torch.dtype, device: torch.device, outdir: Path, prefix: str):
    t_gpu = torch.empty(n_elements, dtype=dtype, device=device).normal_()
    path = outdir / f"{prefix}_{n_elements}_{dtype}_{next(tempfile._get_candidate_names())}.pt"

    # 预热
    _ = time_gpu_to_ssd(t_gpu, path)
    _ = time_ssd_to_gpu(path, device)

    # 正式
    t_write = time_gpu_to_ssd(t_gpu, path)
    t_read, t_back = time_ssd_to_gpu(path, device)

    assert t_back.dtype == dtype and t_back.numel() == n_elements
    nbytes = n_elements * dtype_nbytes(dtype)

    os.remove(path)

    return {
        "mode": "ssd",
        "file": str(path),
        "elements": n_elements,
        "dtype": str(dtype).replace("torch.", ""),
        "tensor_bytes": nbytes,
        "write_sec": t_write,
        "read_sec": t_read,
        "write_GBps": (nbytes / (1024**3)) / t_write if t_write > 0 else float("inf"),
        "read_GBps": (nbytes / (1024**3)) / t_read if t_read > 0 else float("inf"),
    }


def run_g2c_once(n_elements: int, dtype: torch.dtype, device: torch.device, pinned: bool):
    t_gpu = torch.empty(n_elements, dtype=dtype, device=device).normal_()

    # 预热
    _ = time_gpu_to_cpu_only(t_gpu, make_pinned=pinned)

    wall_sec, cuda_sec = time_gpu_to_cpu_only(t_gpu, make_pinned=pinned)
    nbytes = n_elements * dtype_nbytes(dtype)
    return {
        "mode": f"g2c_{'pinned' if pinned else 'pageable'}",
        "elements": n_elements,
        "dtype": str(dtype).replace("torch.", ""),
        "tensor_bytes": nbytes,
        "wall_sec": wall_sec,
        "cuda_sec": cuda_sec,
        "wall_GBps": (nbytes / (1024**3)) / wall_sec if wall_sec > 0 else float("inf"),
        "cuda_GBps": (nbytes / (1024**3)) / cuda_sec if cuda_sec > 0 else float("inf"),
    }


def run_c2g_once(n_elements: int, dtype: torch.dtype, device: torch.device, pinned: bool):
    # 预热
    _ = time_cpu_to_gpu_only(n_elements, dtype, device, make_pinned=pinned)

    wall_sec, cuda_sec = time_cpu_to_gpu_only(n_elements, dtype, device, make_pinned=pinned)
    nbytes = n_elements * dtype_nbytes(dtype)
    return {
        "mode": f"c2g_{'pinned' if pinned else 'pageable'}",
        "elements": n_elements,
        "dtype": str(dtype).replace("torch.", ""),
        "tensor_bytes": nbytes,
        "wall_sec": wall_sec,
        "cuda_sec": cuda_sec,
        "wall_GBps": (nbytes / (1024**3)) / wall_sec if wall_sec > 0 else float("inf"),
        "cuda_GBps": (nbytes / (1024**3)) / cuda_sec if cuda_sec > 0 else float("inf"),
    }


# --------------------------- main ---------------------------


def main():
    parser = argparse.ArgumentParser(description="GPU<->CPU & GPU<->SSD & Network benchmark (PyTorch)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", default="1e8", help='e.g. "1e8,2e8" or "4x4096x4096"')
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--modes", default="g2c,c2g,ssd,net", help="comma-separated: g2c,c2g,ssd,net")
    parser.add_argument("--pinned", action="store_true", help="Use pinned memory for CPU buffers (recommended).")
    parser.add_argument("--outdir", default="./ssd_test", help="SSD mount for ssd mode")
    parser.add_argument("--csv", default="log/res.csv", help="Optional CSV path")

    # 网络相关
    parser.add_argument(
        "--net-role",
        default="loopback",
        choices=["loopback", "server", "client"],
        help="网络模式：单机 loopback，或手动 server/client",
    )
    parser.add_argument("--net-host", default="127.0.0.1")
    parser.add_argument("--net-port", type=int, default=50007)
    parser.add_argument("--net-chunk-bytes", type=int, default=4 * 1024 * 1024, help="发送分片大小")
    parser.add_argument("--net-rate-limit-mbps", type=float, default=0.0, help="发送端限速（Mbps，0=不限速）")
    parser.add_argument("--net-gpu-path", action="store_true", help="网络计时包含 GPU->CPU 和 CPU->GPU（更贴近端到端）")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map[args.dtype]
    sizes = parse_sizes(args.sizes)
    modes = set([m.strip().lower() for m in args.modes.split(",")])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}, dtype: {dtype}, sizes(elements): {sizes}, modes: {sorted(modes)}")
    if "ssd" in modes:
        print("NOTE (ssd): Results include serialization + OS page cache effects.")
    if args.pinned:
        print("Pinned memory: ENABLED (CPU buffers)")

    # 启动 loopback echo server（如需要）
    stop_event = threading.Event()
    server_thread = None
    if "net" in modes and args.net_role in ("loopback", "server"):
        server_thread = EchoServer(args.net_host, args.net_port, stop_event)
        server_thread.start()
        # 简短等待，确保监听就绪
        time.sleep(0.2)
        print(f"[NET] Echo server started on {args.net_host}:{args.net_port} (role={args.net_role})")
        if args.net_role == "server":
            # 仅作为服务器，不做 client 侧测试
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                stop_event.set()
                server_thread.join(timeout=1.0)
            return

    all_rows = []

    for n in sizes:
        # SSD
        if "ssd" in modes:
            res = []
            for r in range(args.repeats):
                row = run_ssd_once(n, dtype, device, outdir, prefix=f"ssd_r{r}")
                res.append(row)
                print(
                    f"[SSD | {n} elems | {human_bytes(row['tensor_bytes'])}] "
                    f"write {row['write_sec']*1000:.1f} ms ({row['write_GBps']:.2f} GB/s) | "
                    f"read {row['read_sec']*1000:.1f} ms ({row['read_GBps']:.2f} GB/s) -> {os.path.basename(row['file'])}"
                )
            med = {
                "mode": "ssd_median",
                "elements": n,
                "dtype": str(dtype).replace("torch.", ""),
                "tensor_bytes": res[0]["tensor_bytes"],
                "write_sec_med": median([x["write_sec"] for x in res]),
                "read_sec_med": median([x["read_sec"] for x in res]),
            }
            med["write_GBps_med"] = (med["tensor_bytes"] / (1024**3)) / med["write_sec_med"]
            med["read_GBps_med"] = (med["tensor_bytes"] / (1024**3)) / med["read_sec_med"]
            all_rows.append(med)
            print(
                f"==> SSD MEDIAN [{n}]: write {med['write_sec_med']*1000:.1f} ms ({med['write_GBps_med']:.2f} GB/s) | "
                f"read {med['read_sec_med']*1000:.1f} ms ({med['read_GBps_med']:.2f} GB/s)"
            )
            print("-" * 80)

        # G2C
        if "g2c" in modes:
            res = []
            for r in range(args.repeats):
                row = run_g2c_once(n, dtype, device, pinned=args.pinned)
                res.append(row)
                print(
                    f"[G2C | {row['mode']} | {n} elems | {human_bytes(row['tensor_bytes'])}] "
                    f"wall {row['wall_sec']*1000:.1f} ms ({row['wall_GBps']:.2f} GB/s) | "
                    f"cuda {row['cuda_sec']*1000:.1f} ms ({row['cuda_GBps']:.2f} GB/s)"
                )
            med = {
                "mode": f"g2c_{'pinned' if args.pinned else 'pageable'}_median",
                "elements": n,
                "dtype": str(dtype).replace("torch.", ""),
                "tensor_bytes": res[0]["tensor_bytes"],
                "wall_sec_med": median([x["wall_sec"] for x in res]),
                "cuda_sec_med": median([x["cuda_sec"] for x in res]),
            }
            med["wall_GBps_med"] = (med["tensor_bytes"] / (1024**3)) / med["wall_sec_med"]
            med["cuda_GBps_med"] = (med["tensor_bytes"] / (1024**3)) / med["cuda_sec_med"]
            all_rows.append(med)
            print(
                f"==> G2C MEDIAN [{n}]: wall {med['wall_sec_med']*1000:.1f} ms ({med['wall_GBps_med']:.2f} GB/s) | "
                f"cuda {med['cuda_sec_med']*1000:.1f} ms ({med['cuda_GBps_med']:.2f} GB/s)"
            )
            print("-" * 80)

        # C2G
        if "c2g" in modes:
            res = []
            for r in range(args.repeats):
                row = run_c2g_once(n, dtype, device, pinned=args.pinned)
                res.append(row)
                print(
                    f"[C2G | {row['mode']} | {n} elems | {human_bytes(row['tensor_bytes'])}] "
                    f"wall {row['wall_sec']*1000:.1f} ms ({row['wall_GBps']:.2f} GB/s) | "
                    f"cuda {row['cuda_sec']*1000:.1f} ms ({row['cuda_GBps']:.2f} GB/s)"
                )
            med = {
                "mode": f"c2g_{'pinned' if args.pinned else 'pageable'}_median",
                "elements": n,
                "dtype": str(dtype).replace("torch.", ""),
                "tensor_bytes": res[0]["tensor_bytes"],
                "wall_sec_med": median([x["wall_sec"] for x in res]),
                "cuda_sec_med": median([x["cuda_sec"] for x in res]),
            }
            med["wall_GBps_med"] = (med["tensor_bytes"] / (1024**3)) / med["wall_sec_med"]
            med["cuda_GBps_med"] = (med["tensor_bytes"] / (1024**3)) / med["cuda_sec_med"]
            all_rows.append(med)
            print(
                f"==> C2G MEDIAN [{n}]: wall {med['wall_sec_med']*1000:.1f} ms ({med['wall_GBps_med']:.2f} GB/s) | "
                f"cuda {med['cuda_sec_med']*1000:.1f} ms ({med['cuda_GBps_med']:.2f} GB/s)"
            )
            print("-" * 80)

        # NET（客户端侧）
        if "net" in modes and args.net_role in ("loopback", "client"):
            res = []
            for r in range(args.repeats):
                row = run_net_once(
                    n,
                    dtype,
                    device,
                    host=args.net_host,
                    port=args.net_port,
                    chunk_bytes=args.net_chunk_bytes,
                    rate_limit_mbps=(args.net_rate_limit_mbps or None),
                    include_gpu_path=args.net_gpu_path,
                )
                res.append(row)
                print(
                    f"[NET | {row['mode']} | {n} elems | wire {human_bytes(row['wire_bytes'])} | "
                    f"rate_limit={args.net_rate_limit_mbps} Mbps | chunk={human_bytes(args.net_chunk_bytes)}] "
                    f"RTT {row['rtt_sec']*1000:.1f} ms | est one-way {row['est_oneway_GBps']:.2f} GB/s"
                )
            med = {
                "mode": f"net_{'gpu' if args.net_gpu_path else 'cpu'}_median",
                "elements": n,
                "dtype": str(dtype).replace("torch.", ""),
                "tensor_bytes": n * dtype_nbytes(dtype),
                "wire_bytes_med": median([x["wire_bytes"] for x in res]),
                "rtt_sec_med": median([x["rtt_sec"] for x in res]),
                "est_oneway_GBps_med": median([x["est_oneway_GBps"] for x in res]),
                "rate_limit_Mbps": args.net_rate_limit_mbps or 0.0,
                "chunk_bytes": args.net_chunk_bytes,
            }
            all_rows.append(med)
            print(f"==> NET MEDIAN [{n}]: RTT {med['rtt_sec_med']*1000:.1f} ms | " f"est one-way {med['est_oneway_GBps_med']:.2f} GB/s")
            print("-" * 80)

    if args.csv and all_rows:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"Saved CSV to: {csv_path.resolve()}")

    # 优雅关闭服务器
    if server_thread is not None:
        stop_event.set()
        # 触发 accept 退出：短暂连接一下
        try:
            with socket.create_connection((args.net_host, args.net_port), timeout=1):
                pass
        except Exception:
            pass
        server_thread.join(timeout=1.0)

    print("Done.")


if __name__ == "__main__":
    main()
