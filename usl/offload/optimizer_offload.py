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
        self.except_tensor_idx_list = except_tensor_idx_list  # list of tensor index to be excluded from offload/reload
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
                    if v.device.type == 'cuda':  # move tensor to CPU memory
                        t_cpu = torch.empty_like(v, device='cpu', pin_memory=True)
                        t_cpu.data.copy_(v.data, non_blocking=True)
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
                    if v.device.type == 'cpu':  # move tensor to GPU memory
                        t_gpu = torch.empty_like(v, device=self.device)
                        t_gpu.data.copy_(v.data, non_blocking=True)
                        v.data = t_gpu.data
            pass

    # offload optimizer state from GPU to CPU
    def offload(self, async_offload=False):
        stream = self.offload_stream if self.offload_stream else torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            self._offload()

        if async_offload:
            # record offload event
            self.offload_event.record(stream)
        else:
            # wait for all optimizer state offloaded
            torch.cuda.synchronize(self.device)
            # release GPU memory for optimizer state

    # wait for all optimizer state offloaded to finish
    def wait_offload(self):
        if self.offload_stream != torch.cuda.current_stream(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.offload_event)

    # reload optimizer state from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            self._reload()

        if async_reload:
            # record reload event
            self.reload_event.record(stream)
        else:
            # wait for all optimizer states reloaded
            torch.cuda.synchronize(self.device)
            # release CPU memory for optimizer states
            torch.cuda.empty_cache()

    def wait_reload(self):
        if self.load_stream != torch.cuda.current_stream(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.reload_event)
            # self.optimizer_state_on_cpu.clear()
            torch.cuda.empty_cache()
