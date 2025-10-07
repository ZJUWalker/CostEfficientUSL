import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class OptimizerStateOffload:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        offload_threshold: int = 0,
        device='cuda',
        load_stream=None,
        offload_stream=None,
        except_tensor_idx_list: List[int] = [],
    ):
        self.optimizer = optimizer
        self.offload_threshold = offload_threshold  # Byte
        self.device = device
        self.load_stream = load_stream if load_stream else torch.cuda.Stream(device)  # create a new stream for offload/reload
        self.offload_stream = offload_stream if offload_stream else torch.cuda.Stream(device)  # create a new stream for offload/reload
        self.compute_stream = torch.cuda.current_stream(device)  # use current stream for compute
        self.except_tensor_idx_list = except_tensor_idx_list  # list of tensor index to be excluded from offload/reload
        self.optimizer_state_on_cpu: Dict[torch.Tensor, Dict[str, torch.Tensor]] = {}  # param -> parameter state on CPU DRAM
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)

    def _offload(self):
        for param, state in self.optimizer.state.items():
            if id(param) in self.except_tensor_idx_list:
                print(f"Skip offload for tensor {id(param)}")
                continue
            state: Dict[str, torch.Tensor]
            for k, v in state.items():
                v: torch.Tensor
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.numel() * v.element_size() >= self.offload_threshold:
                    if v.is_cuda:  # move tensor to CPU memory
                        t_cpu = torch.empty_like(v, device='cpu', pin_memory=True)
                        t_cpu.copy_(v, non_blocking=True)
                        v.data = t_cpu.data
            pass

    def _reload(self):
        for param, state in self.optimizer.state.items():
            if id(param) in self.except_tensor_idx_list:
                print(f"Skip tensor {id(param)}")
                continue
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
        with torch.cuda.stream(stream):
            self._offload()

        if async_offload:
            # record offload event
            self.offload_event.record(stream)
        else:
            # wait for all optimizer state offloaded
            self.compute_stream.wait_stream(stream)
            # release GPU memory for optimizer state

    # wait for all optimizer state offloaded to finish
    def wait_offload(self):
        if self.offload_stream != self.compute_stream:
            self.compute_stream.wait_event(self.offload_event)

    # reload optimizer state from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # reload should be done after compute
        with torch.cuda.stream(stream):
            self._reload()

        if async_reload:
            # record reload event
            self.reload_event.record(stream)
        else:
            # wait for all optimizer states reloaded
            self.compute_stream.wait_stream(stream)

    def wait_reload(self):
        if self.load_stream != self.compute_stream:
            self.compute_stream.wait_event(self.reload_event)

    def offload_finished(self):
        return self.offload_event.query()

    def reload_finished(self):
        return self.reload_event.query()
