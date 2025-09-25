import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class ModelParamOffload:

    def __init__(self, base_model: nn.Module, offload_threshold: int = 0, device='cuda', swap_stream=None):
        self.base_model = base_model
        self.offload_threshold = offload_threshold  # Byte
        self.device = device
        self.model_param_on_gpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> parameter state on GPU
        self.model_param_on_cpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> parameter state on CPU DRAM
        self.swap_stream = torch.cuda.Stream(device) if swap_stream is None else swap_stream
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)
        self._init_param_on_cpu()

    def _init_param_on_cpu(self):
        for name, param in self.base_model.named_parameters():
            self.model_param_on_cpu[id(param)] = param
            self.model_param_on_gpu[id(param)] = param  # use the same tensor for both on CPU and GPU,temporary

    # offload model parameters and optimizer states from GPU to CPU
    def offload(self, async_offload=False):
        stream = self.swap_stream if self.swap_stream else torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            # Offload model parameters to CPU
            for name, tensor in self.model_param_on_gpu.items():
                if tensor.numel() * tensor.element_size() > self.offload_threshold:
                    t_cpu = torch.empty_like(tensor, device='cpu', pin_memory=True)
                    t_cpu.data.copy_(tensor.data, non_blocking=True)
                    self.model_param_on_cpu[id(tensor)] = t_cpu

            if async_offload:
                # record offload event
                self.offload_event.record(stream)
            else:
                # wait for all tensors offloaded
                torch.cuda.synchronize(self.device)
                # release GPU memory
                self._release_gpu_memory()

    # wait for all offloaded states to finish
    def wait_offload(self):
        if self.swap_stream != torch.cuda.current_stream(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.offload_event)
            self._release_gpu_memory()

    def _release_gpu_memory(self):
        # Release model parameters and optimizer states from GPU
        for idx in self.model_param_on_gpu.keys():
            self.model_param_on_gpu[idx].data = torch.empty(0)
        torch.cuda.empty_cache()

    # reload model parameters and optimizer states from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.swap_stream if self.swap_stream else torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            # Reload model parameters from CPU to GPU
            for idx, tensor in self.model_param_on_cpu.items():
                t_gpu = torch.empty_like(tensor, device=self.device)  # no pin_memory on GPU
                t_gpu.data.copy_(tensor.data, non_blocking=True)
                self.model_param_on_gpu[idx].data = t_gpu.data
            if async_reload:
                # record reload event
                self.reload_event.record(stream)
            else:
                # wait for all tensors reloaded
                torch.cuda.synchronize(self.device)
                # release CPU memory
                self.model_param_on_cpu.clear()
                torch.cuda.empty_cache()

    def wait_reload(self):
        if self.swap_stream != torch.cuda.current_stream(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.reload_event)
            # Clear CPU memory
            self.model_param_on_cpu.clear()
            torch.cuda.empty_cache()
