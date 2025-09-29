import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Union
from torch.autograd.graph import saved_tensors_hooks


class ContiguousActivationOffload:
    """
    Offload autograd saved tensors to CPU using torch.autograd.graph.saved_tensors_hooks,
    with asynchronous GPU<->CPU transfers via CUDA streams & events.

    Args:
        device: 反传时搬回的目标设备。None 表示按原始张量 device。
        pin_memory: CPU 端是否使用页锁内存，便于非阻塞 H2D 传输。
        threshold_bytes: 仅 offload 大于该阈值（字节）的张量，默认 0 表示全部 offload。
        cpu_dtype: 在 CPU 端保存用的 dtype（可压缩以省内存，可能带来轻微数值误差）。None 表示保持原 dtype。
        collect_stats: 是否记录统计。
        use_reload_stream: 是否为 H2D 使用独立流（否则用当前流）。
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        pin_memory: bool = True,
        offload_threshold: int = 0,
        cpu_dtype: Optional[torch.dtype] = torch.float32,
        collect_stats: bool = True,
        async_offload: bool = True,
        offload_stream: Optional[torch.cuda.Stream] = None,
        reload_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.device = device
        self.pin_memory = bool(pin_memory)
        self.threshold_bytes = int(offload_threshold)
        self.cpu_dtype = cpu_dtype
        self.collect_stats = bool(collect_stats)
        self.async_offload = bool(async_offload)
        # init streams and events
        self._offload_stream: Optional[torch.cuda.Stream] = offload_stream if offload_stream else torch.cuda.Stream(device)
        self._reload_stream: Optional[torch.cuda.Stream] = reload_stream if reload_stream else torch.cuda.Stream(device)
        # init events for synchronization and profile
        self.offload_start_event = torch.cuda.Event(enable_timing=True)
        self.offload_end_event = torch.cuda.Event(enable_timing=True)
        self.reload_start_event = torch.cuda.Event(enable_timing=True)
        self.reload_end_event = torch.cuda.Event(enable_timing=True)

        self._context_manager = None

        self.reset_stats()

    # --- public: stats ---
    def reset_stats(self):
        self.stats = {
            "status": "RELOAD",
            "num_saved": 0,
            "num_offloaded": 0,
            "bytes_offloaded": 0,
        }

    # --- helpers ---
    def _nbytes(self, t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def _should_offload(self, t: torch.Tensor) -> bool:
        if not t.is_cuda:
            return False
        return self._nbytes(t) > self.threshold_bytes

    def wait_offload(self):
        if self.async_offload:
            torch.cuda.current_stream().wait_event(self.offload_end_event)

    def wait_reload(self):
        if self.async_offload:
            torch.cuda.current_stream().wait_event(self.reload_end_event)

    def __enter__(self):
        def pack(x: torch.Tensor):
            # hooks 理论上只会进张量，保险起见做下判断
            if not isinstance(x, torch.Tensor):
                return x

            if self.collect_stats:
                if self.stats["status"] == "RELOAD":
                    self.stats["status"] = "OFFLOAD"
                    self.offload_start_event.record()
                self.stats["num_saved"] += 1

            if not self._should_offload(x):
                return x  # 小张量不搬

            dev = x.device
            assert dev.type == "cuda", "should_offload 只在 CUDA 张量时返回 True"

            # 1) 在 offload_stream 上发起 D2H 异步拷贝
            if self.cpu_dtype is None:
                save_dtype = x.dtype
            else:
                save_dtype = self.cpu_dtype

            if self.async_offload:
                with torch.cuda.stream(self._offload_stream):
                    # pinned CPU buffer
                    cpu_buf = torch.empty_like(x, device="cpu", dtype=save_dtype, pin_memory=self.pin_memory)
                    # 注意：non_blocking 依赖 pinned memory
                    cpu_buf.copy_(x, non_blocking=True)
                    # 记录完成事件（仍是入队，不阻塞）
                    self.offload_end_event.record(self._offload_stream)
            else:
                cpu_buf = torch.empty_like(x, device="cpu", dtype=save_dtype, pin_memory=self.pin_memory)
                # 注意：non_blocking 依赖 pinned memory
                cpu_buf.copy_(x, non_blocking=False)

            if self.collect_stats:
                nb = self._nbytes(x)
                self.stats["num_offloaded"] += 1
                self.stats["bytes_offloaded"] += nb

            # 将必要元数据打包，供 unpack 使用
            meta = (cpu_buf, x.dtype, dev)
            return meta

        def unpack(obj):
            # wait for offload_stream to complete
            if self.async_offload:
                if self.stats["status"] == "OFFLOAD":
                    self.stats["status"] = "RELOAD"
                    self.reload_start_event.record()
                self._reload_stream.wait_event(self.offload_end_event)
            # 未被我们打包过的，直接返回（比如小张量）
            if isinstance(obj, torch.Tensor):
                return obj

            cpu_buf, orig_dtype, orig_device = obj
            cpu_buf: torch.Tensor

            # 确定目标设备
            if self.device is not None:
                target_dev = self.device
            else:
                target_dev = (
                    orig_device
                    if isinstance(orig_device, torch.device)
                    else (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))
                )
            # 同步 H2D 拷贝
            t_gpu = torch.empty_like(cpu_buf, device=target_dev, dtype=orig_dtype)
            t_gpu.copy_(cpu_buf, non_blocking=True)
            # 记录完成事件（仍是入队，不阻塞）
            self.reload_end_event.record(self._reload_stream)
            torch.cuda.current_stream().wait_event(self.reload_end_event)

            # 重要：不做 synchronize。让依赖通过流/事件衔接，调用方在自己的当前流上
            # 若使用独立 reload_stream，返回的 t_gpu 在该流上有未完成的拷贝；
            # CUDA 语义保证后续在当前流上的使用会自动同步（不同流间按需插入依赖）。
            return t_gpu

        self._context_manager = saved_tensors_hooks(pack, unpack)
        self._context_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._context_manager is not None:
            self._context_manager.__exit__(exc_type, exc, tb)
            self._context_manager = None
            torch.cuda.synchronize()
