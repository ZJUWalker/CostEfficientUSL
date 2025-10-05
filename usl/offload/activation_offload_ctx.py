import torch
import torch.nn as nn
from typing import Any, List, Optional, Tuple, Dict, Union
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
        mini_batch_num: int = 1,
        device: Optional[torch.device] = None,
        pin_memory: bool = True,
        offload_threshold: int = 0,
        cpu_dtype: Optional[torch.dtype] = torch.float32,
        collect_stats: bool = True,
        async_offload: bool = True,
        offload_stream: Optional[torch.cuda.Stream] = None,
        reload_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.mini_batch_num = mini_batch_num
        self.curr_batch_idx = 0
        self.device = device
        self.pin_memory = bool(pin_memory)
        self.threshold_bytes = int(offload_threshold)
        self.cpu_dtype = cpu_dtype
        self.collect_stats = bool(collect_stats)
        self.async_offload = bool(async_offload)
        # init streams and events
        self._offload_stream: Optional[torch.cuda.Stream] = (
            offload_stream if offload_stream else torch.cuda.Stream(device)
        )

        self._reload_stream: Optional[torch.cuda.Stream] = reload_stream if reload_stream else torch.cuda.Stream(device)
        # init events for synchronization and profile
        self.offload_start_event = torch.cuda.Event(enable_timing=True)
        self.offload_end_event = torch.cuda.Event(enable_timing=True)
        self.reload_start_event = torch.cuda.Event(enable_timing=True)
        self.reload_end_event = torch.cuda.Event(enable_timing=True)

        # init tensors list for offloading / reloading
        self.tensors_to_offload: List[Dict] = []

        self._context_manager = None

        self.reset_stats()

    def step(self):
        self.curr_batch_idx += 1
        if self.curr_batch_idx == self.mini_batch_num:
            self.curr_batch_idx = 0
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
        if not t.is_cuda or not t.requires_grad:
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
            if not isinstance(x, torch.Tensor) or not x.is_cuda or not x.requires_grad:
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

            if self.collect_stats:
                nb = self._nbytes(x)
                self.stats["num_offloaded"] += 1
                self.stats["bytes_offloaded"] += nb
            if x.data_ptr() in self.tensors_to_offload[self.curr_batch_idx]:
                self.tensors_to_offload[self.curr_batch_idx][x.data_ptr()] = x

            # 将必要元数据打包，供 unpack 使用
            return (self.curr_batch_idx, x.data_ptr())

        def unpack(obj: Tuple[Any]):
            # 未被我们打包过的，直接返回（比如小张量）
            if isinstance(obj, torch.Tensor):
                return obj

            (curr_batch_idx, tensor_idx) = obj
            if curr_batch_idx == 0:
                self.stats["status"] = "RELOAD"
                self.reload_start_event.record()

            return None

        self._context_manager = saved_tensors_hooks(pack, unpack)
        self._context_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._context_manager is not None:
            self._context_manager.__exit__(exc_type, exc, tb)
            self._context_manager = None
            torch.cuda.synchronize()
