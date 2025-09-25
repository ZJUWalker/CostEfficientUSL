import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class OptimizerStateOffload:

    def __init__(self, optimizer: torch.optim.Optimizer, offload_threshold: int = 0, device='cuda', swap_stream=None):
        self.optimizer = optimizer
        self.offload_threshold = offload_threshold  # Byte
        self.device = device
        self.curr_optimizer_state = 'cpu'  # 'cpu' or 'cuda'
        self.swap_stream = swap_stream if swap_stream else torch.cuda.Stream(device)  # create a new stream for offload/reload
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)

    def _offload(self):
        for param, state in self.optimizer.state.items():
            state: Dict[str, torch.Tensor]
            for k, v in state.items():
                v: torch.Tensor
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    if v.device.type == 'cuda':  # move tensor to CPU memory
                        t_cpu = torch.empty_like(v, device='cpu', pin_memory=True)
                        t_cpu.data.copy_(v.data, non_blocking=True)
                        v.data = t_cpu.data
            pass

    def _reload(self):
        for param, state in self.optimizer.state.items():
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
        if self.curr_optimizer_state == 'cpu':
            return  # no need to offload again
        stream = self.swap_stream if self.swap_stream else torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            self._offload()

        if async_offload:
            # record offload event
            self.offload_event.record(stream)
        else:
            # wait for all optimizer state offloaded
            torch.cuda.synchronize(self.device)
            # release GPU memory for optimizer state
            self.curr_optimizer_state = 'cpu'

    # wait for all optimizer state offloaded to finish
    def wait_offload(self):
        if self.swap_stream != torch.cuda.current_stream(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.offload_event)
            self.curr_optimizer_state = 'cpu'

    # reload optimizer state from CPU to GPU
    def reload(self, async_reload=False):
        # if not self.inited:
        # self._lazy_init_optimizer_state_dict()
        if self.curr_optimizer_state == 'cuda':
            return  # no need to reload again
        stream = self.swap_stream if self.swap_stream else torch.cuda.current_stream(self.device)
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
            self.curr_optimizer_state = 'cuda'

    def wait_reload(self):
        if self.swap_stream != torch.cuda.current_stream(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.reload_event)
            # self.optimizer_state_on_cpu.clear()
            torch.cuda.empty_cache()
            self.curr_optimizer_state = 'cuda'
