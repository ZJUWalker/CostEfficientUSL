import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import time


class OptimizerStateOffload:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        offload_threshold: int = 0,
        # offload_ratio: float = 1.0,
        offload_until_param_id: int = -1,
        device='cuda',
        load_stream=None,
        offload_stream=None,
        except_tensor_idx_list: List[int] = [],
    ):
        self.optimizer = optimizer
        self.offload_threshold = offload_threshold  # Byte
        # self.offload_ratio = float(max(0.0, min(1.0, offload_ratio)))  # ratio of offload
        self.offload_until_param_id = offload_until_param_id  # id of the last parameter to be offloaded
        self.device = device
        self.load_stream = load_stream if load_stream else torch.cuda.Stream(device)  # create a new stream for offload/reload
        self.offload_stream = offload_stream if offload_stream else torch.cuda.Stream(device)  # create a new stream for offload/reload
        self.compute_stream = torch.cuda.current_stream(device)  # use current stream for compute
        self.except_tensor_idx_list = except_tensor_idx_list  # list of tensor index to be excluded from offload/reload
        self.optimizer_state_on_cpu: Dict[torch.Tensor, Dict[str, torch.Tensor]] = {}  # param -> parameter state on CPU DRAM
        self.start_offload_event = torch.cuda.Event(enable_timing=True)
        self.start_reload_event = torch.cuda.Event(enable_timing=True)
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)
        # profile events
        self.offload_timestamp = [0, 0]
        self.reload_timestamp = [0, 0]

        # 统计用途：最近一次offload实际卸载的字节数
        # self._last_offloaded_bytes = 0

    # def _iter_candidate_tensors(self):
    #     """
    #     遍历 optimizer.state，产出可被考虑 offload 的 (tensor, bytes)。
    #     只产出：是 Tensor、维度>0、在 GPU 上、大小>=threshold、且不在排除名单中的条目。
    #     """
    #     for param, state in self.optimizer.state.items():
    #         if id(param) in self.except_tensor_idx_list:
    #             # print(f"Skip offload for tensor {id(param)}")
    #             continue
    #         state: Dict[str, torch.Tensor]
    #         for k, v in state.items():
    #             if not isinstance(v, torch.Tensor):
    #                 continue
    #             if v.dim() <= 0:
    #                 continue
    #             if not v.is_cuda:
    #                 # 已在CPU上的（之前卸载过），这次不作为 offload 候选
    #                 continue
    #             size_bytes = v.numel() * v.element_size()
    #             self.total_os_param_count += v.numel()
    #             self.os_tensor_start_list.append(self.total_os_param_count)
    #             if size_bytes < self.offload_threshold:
    #                 continue
    #             yield v, size_bytes

    # def _offload(self):
    #     """
    #     按 offload_ratio 卸载一部分状态：
    #     - ratio == 0: 不卸载
    #     - ratio == 1: 卸载所有候选
    #     - 其余：按大小从大到小卸载，直到达到目标字节数
    #     """
    #     if self.offload_ratio <= 0.0:
    #         self._last_offloaded_bytes = 0
    #         return

    #     # 收集候选
    #     candidates: List[Tuple[torch.Tensor, int]] = list(self._iter_candidate_tensors())
    #     if not candidates:
    #         self._last_offloaded_bytes = 0
    #         return

    #     total_bytes = sum(sz for _, sz in candidates)
    #     target_bytes = int(total_bytes * self.offload_ratio)

    #     # 全卸载的捷径
    #     if self.offload_ratio >= 0.999999:
    #         selected = candidates
    #     else:
    #         # 按大小从大到小排序，优先卸载大块，释放显存更高效
    #         # candidates.sort(key=lambda x: x[1], reverse=True)
    #         selected = []
    #         acc = 0
    #         for v, sz in candidates:
    #             selected.append((v, sz))
    #             acc += sz
    #             if acc >= target_bytes:
    #                 break

    #     # 执行卸载：把选中的 GPU Tensor 复制到 pinned CPU，再替换 data 指针
    #     actually_offloaded = 0
    #     for v, sz in selected:
    #         # 再次防御性检查（并行场景下可能发生状态变化）
    #         if not v.is_cuda:
    #             continue
    #         t_cpu = torch.empty_like(v, device='cpu', pin_memory=True)
    #         t_cpu.copy_(v, non_blocking=True)
    #         v.data = t_cpu.data
    #         actually_offloaded += sz
    #     # print(f"offload ratio {self.offload_ratio}, offload {len(selected)} tensors, actually offload {actually_offloaded} bytes")

    #     self._last_offloaded_bytes = actually_offloaded

    # @property
    # def last_offloaded_bytes(self) -> int:
    #     """上次 offload 实际从 GPU 卸载掉的字节数（统计值）。"""
    #     return self._last_offloaded_bytes

    def _offload(self):
        total_offload_bytes = 0
        for param, state in self.optimizer.state.items():
            if id(param) in self.except_tensor_idx_list:
                print(f"Skip offload for tensor {id(param)}")
                continue
            if id(param) == self.offload_until_param_id:
                # TODO: if the model is wrapped by LoRA, we need to offload all of the model parameters and the optimizer state
                # print(f"Stop offloading, reach {id(param)},total offload bytes {total_offload_bytes}")
                break
            state: Dict[str, torch.Tensor]
            for k, v in state.items():
                v: torch.Tensor
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.numel() * v.element_size() >= self.offload_threshold:
                    if v.is_cuda:  # move tensor to CPU memory
                        t_cpu = torch.empty_like(v, device='cpu', pin_memory=True)
                        t_cpu.copy_(v, non_blocking=True)
                        v.data = t_cpu.data
                        total_offload_bytes += v.numel() * v.element_size()
            pass

    def _reload(self):
        for param, state in self.optimizer.state.items():
            if id(param) in self.except_tensor_idx_list:
                print(f"Skip tensor {id(param)}")
                continue
            # if id(param) == self.offload_until_param_id:
            #     break
            state: Dict[str, torch.Tensor]
            for k, v in state.items():
                v: torch.Tensor
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    if v.is_cpu:  # move tensor to GPU memory
                        t_gpu = torch.empty_like(v, device=self.device)
                        t_gpu.copy_(v, non_blocking=True)
                        v.data = t_gpu.data
            pass

    # offload optimizer state from GPU to CPU
    def offload(self, async_offload=False):
        stream = self.offload_stream if self.offload_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # offload should be done after compute
        # record start offload event used for profiling
        self.start_offload_event.record(stream)
        self.offload_timestamp[0] = time.perf_counter()

        with torch.cuda.stream(stream):
            self._offload()

        if async_offload:
            # record offload event
            self.offload_event.record(stream)
        else:
            # wait for all optimizer state offloaded
            self.compute_stream.wait_stream(stream)
            self.offload_timestamp[1] = time.perf_counter()
            # release GPU memory for optimizer state

    # wait for all optimizer state offloaded to finish
    def wait_offload(self):
        if self.offload_stream != self.compute_stream:
            self.compute_stream.wait_event(self.offload_event)
            self.offload_event.synchronize()
            elapsed_time = self.start_offload_event.elapsed_time(self.offload_event)  # kernel time in ms
            self.offload_timestamp[1] = self.offload_timestamp[0] + elapsed_time / 1000  # time in seconds
        return self.offload_timestamp

    # reload optimizer state from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # reload should be done after compute
        # used for profiling
        self.start_reload_event.record(stream)
        self.reload_timestamp[0] = time.perf_counter()
        with torch.cuda.stream(stream):
            self._reload()

        if async_reload:
            # record reload event
            self.reload_event.record(stream)
        else:
            # wait for all optimizer states reloaded
            self.compute_stream.wait_stream(stream)
            self.reload_timestamp[1] = time.perf_counter()

    def wait_reload(self):
        if self.load_stream != self.compute_stream:
            self.compute_stream.wait_event(self.reload_event)
            # used for profiling
            self.reload_event.synchronize()
            elapsed_time = self.start_reload_event.elapsed_time(self.reload_event)  # kernel time in ms
            self.reload_timestamp[1] = self.reload_timestamp[0] + elapsed_time / 1000  # time in seconds
        return self.reload_timestamp

    def offload_finished(self):
        return self.offload_event.query()

    def reload_finished(self):
        return self.reload_event.query()
