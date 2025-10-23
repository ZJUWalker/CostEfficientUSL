import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import time


class ModelParamOffload:

    def __init__(
        self,
        base_model: nn.Module,
        offload_threshold: int = 0,
        # offload_ratio: float = 1.0,
        offload_layer_num: int = 4,
        device="cuda",
        load_stream=None,
        offload_stream=None,
        except_tensor_idx_list=[],
    ):
        self.base_model = base_model
        self.offload_threshold = offload_threshold  # Byte
        # self.offload_ratio = offload_ratio  # ratio of parameters to offload
        self.offload_layer_num = offload_layer_num
        self.device = device
        self.model_param_on_gpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> parameter state on GPU
        self.model_param_on_cpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> parameter state on CPU DRAM
        # self.swap_stream = torch.cuda.Stream(device) if swap_stream is None else swap_stream
        self.load_stream: torch.cuda.Stream = load_stream
        self.offload_stream: torch.cuda.Stream = offload_stream
        self.compute_stream: torch.cuda.Stream = torch.cuda.current_stream(self.device)
        # def events
        self.start_offload_event = torch.cuda.Event(enable_timing=True)
        self.start_reload_event = torch.cuda.Event(enable_timing=True)
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)
        # profile events
        self.offload_timestamp = [0, 0]
        self.reload_timestamp = [0, 0]
        self.except_tensor_idx_list: List[int] = except_tensor_idx_list  # tensor_idx list to be excluded from offloading
        self.offload_until_param_id: int = -1
        self.param_count = self._get_param_count()
        # self.max_offload_param_count = self.param_count * self.offload_ratio
        self._init_param_dict()

    def _get_param_count(self):
        count = 0
        for name, param in self.base_model.named_parameters():
            if id(param) in self.except_tensor_idx_list or param.numel() * param.element_size() < self.offload_threshold:
                continue
            count += param.numel()
        return count

    def add_except_tensor(self, tensor: torch.Tensor):
        tensor_idx = id(tensor)
        if tensor_idx not in self.except_tensor_idx_list:
            self.except_tensor_idx_list.append(tensor_idx)

    def remove_except_tensor(self, tensor: torch.Tensor):
        tensor_idx = id(tensor)
        if tensor_idx in self.except_tensor_idx_list:
            self.except_tensor_idx_list.remove(tensor_idx)

    def _init_param_dict(self):
        curr_count = 0
        curr_layer_idx = 0
        for name, param in self.base_model.named_parameters():
            if (
                id(param) in self.except_tensor_idx_list
                or param.numel() * param.element_size() < self.offload_threshold
                # or curr_count >= self.max_offload_param_count
                or self.offload_layer_num == 0  # no offloading
                or curr_layer_idx > self.offload_layer_num
            ):
                param.data = param.data.cuda(self.device)  # pin on GPU
            else:
                if f'h.{curr_layer_idx}' in name or f'layers.{curr_layer_idx}' in name:
                    curr_layer_idx += 1
                    if curr_layer_idx > self.offload_layer_num:
                        self.offload_until_param_id = id(param)
                        param.data = param.data.cuda(self.device)
                        continue
                    # print(name)
                curr_count += param.numel()
                self.model_param_on_cpu[id(param)] = param
                self.model_param_on_gpu[id(param)] = param  # use the same tensor for both on CPU and GPU,temporary
        print(f'offload {curr_count} parameters to CPU, {len(self.model_param_on_cpu)} parameters')

    # offload model parameters and optimizer states from GPU to CPU
    def offload(self, async_offload=False):
        stream = self.offload_stream if self.offload_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # offload should be done after compute
        # record start offload event used for profiling
        self.start_offload_event.record(stream)
        self.start_offload_event.synchronize()
        self.offload_timestamp[0] = time.perf_counter()
        # ----------------------------------------------------
        with torch.cuda.stream(stream):
            # Offload model parameters to CPU
            for idx, tensor in self.model_param_on_gpu.items():
                assert tensor.is_cuda, 'model_param_on_gpu should be on GPU'
                t_cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                t_cpu.data.copy_(tensor.data, non_blocking=True)
                self.model_param_on_cpu[id(tensor)] = t_cpu

            if async_offload:
                # record offload event
                self.offload_event.record(stream)
            else:
                # wait for all tensors offloaded
                self.compute_stream.wait_stream(stream)
                self.offload_timestamp[1] = time.perf_counter()
                # release GPU memory
                self._release_gpu_memory()

    # wait for all offloaded states to finish
    def wait_offload(self):
        if self.offload_stream != self.compute_stream:
            self.compute_stream.wait_event(self.offload_event)
            self.offload_event.synchronize()
            elapsed_time = self.start_offload_event.elapsed_time(self.offload_event)  # kernel time in ms
            self.offload_timestamp[1] = self.offload_timestamp[0] + elapsed_time / 1000  # time in seconds
            self._release_gpu_memory()
        return self.offload_timestamp

    def offload_finished(self):
        return self.offload_event.query()

    def reload_finished(self):
        return self.reload_event.query()

    def _release_gpu_memory(self):
        # Release model parameters and optimizer states from GPU
        for idx in self.model_param_on_gpu.keys():
            if idx in self.except_tensor_idx_list:
                continue
            self.model_param_on_gpu[idx].data = torch.empty(0)
        # torch.cuda.empty_cache()

    # reload model parameters and optimizer states from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # reload should be done after compute
        # used for profiling
        self.start_reload_event.record(stream)
        self.start_reload_event.synchronize()
        self.reload_timestamp[0] = time.perf_counter()
        # ----------------------------------------------------
        with torch.cuda.stream(stream):
            # Reload model parameters from CPU to GPU
            for idx, tensor in self.model_param_on_cpu.items():
                assert tensor.is_cpu and tensor.is_pinned, 'model_param_on_cpu should be on CPU DRAM and pinned'
                t_gpu = torch.empty_like(tensor, device=self.device)  # no pin_memory on GPU
                t_gpu.data.copy_(tensor.data, non_blocking=True)
                self.model_param_on_gpu[idx].data = t_gpu.data
            if async_reload:
                # record reload event
                self.reload_event.record(stream)
            else:
                # wait for all tensors reloaded
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
