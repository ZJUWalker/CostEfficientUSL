import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import time
import sys
from typing_extensions import deprecated


"""
 this is a hook to offload model parameters to device when model params are not captured by autograd.
"""


class ModelParamOffload:

    def __init__(
        self,
        base_model: nn.Module,
        offload_threshold: int = 0,
        # offload_ratio: float = 1.0,
        offload_layer_num: int = 1000,
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

    # def _init_param_dict(self):
    #     curr_count = 0
    #     curr_layer_idx = 0
    #     for name, param in self.base_model.named_parameters():
    #         if (
    #             id(param) in self.except_tensor_idx_list
    #             or param.numel() * param.element_size() < self.offload_threshold
    #             # or curr_count >= self.max_offload_param_count
    #             or self.offload_layer_num == 0  # no offloading
    #             or curr_layer_idx > self.offload_layer_num
    #         ):
    #             param.data = param.data.cuda(self.device)  # pin on GPU
    #         else:
    #             if f'h.{curr_layer_idx}' in name or f'layers.{curr_layer_idx}' in name:
    #                 curr_layer_idx += 1
    #                 if curr_layer_idx > self.offload_layer_num:
    #                     self.offload_until_param_id = id(param)
    #                     param.data = param.data.cuda(self.device)
    #                     continue
    #                 # print(name)
    #             curr_count += param.numel()
    #             self.model_param_on_cpu[id(param)] = param
    #             self.model_param_on_gpu[id(param)] = param  # use the same tensor for both on CPU and GPU,temporary
    #     # print(f'offload {curr_count} parameters to CPU, {len(self.model_param_on_cpu)} parameters')

    def _init_param_dict(self):
        curr_count = 0
        curr_layer_idx = 0

        # 清空 cpu 字典，防止有残留
        self.model_param_on_cpu = {}

        for name, param in self.base_model.named_parameters():
            p_id = id(param)

            # --- 原有的过滤逻辑保持不变 ---
            if (
                p_id in self.except_tensor_idx_list
                or param.numel() * param.element_size() < self.offload_threshold
                or self.offload_layer_num == 0
                or curr_layer_idx > self.offload_layer_num
            ):

                param.data = param.data.cuda(self.device)
                continue

            # --- 层级计数逻辑保持不变 ---
            if f'h.{curr_layer_idx}' in name or f'layers.{curr_layer_idx}' in name:
                curr_layer_idx += 1
                if curr_layer_idx > self.offload_layer_num:
                    self.offload_until_param_id = p_id
                    param.data = param.data.cuda(self.device)
                    continue

            curr_count += param.numel()

            # ==========================================================
            # 【核心修复点】在这里分配独立的 CPU Pinned Memory
            # ==========================================================
            # 错误写法 (导致你报错的原因):
            # self.model_param_on_cpu[p_id] = param

            # 正确写法:
            self.model_param_on_cpu[p_id] = torch.empty(
                param.size(), dtype=param.dtype, layout=param.layout, device='cpu', pin_memory=True  # 必须开启
            )
            self.model_param_on_cpu[p_id].copy_(param.data, non_blocking=False)

            # 记录 GPU 参数引用
            self.model_param_on_gpu[p_id] = param

            # 可选：初始化时先把参数 copy 到 CPU 一份，防止第一次 Offload 前数据不一致
            # self.model_param_on_cpu[p_id].copy_(param.data, non_blocking=True)

        print(f"Initialized Offload Buffers. Total Params Offloaded: {curr_count/1e9:.2f} B")

    # offload model parameters and optimizer states from GPU to CPU
    def offload(self, async_offload=False):
        stream = self.offload_stream if self.offload_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # offload should be done after compute
        # record start offload event used for profiling
        self.start_offload_event.record(stream)
        self.start_offload_event.synchronize()
        self.offload_timestamp[0] = time.time()
        # ----------------------------------------------------
        with torch.cuda.stream(stream):
            # Offload model parameters to CPU
            # assert len(self.model_param_on_gpu) == len(self.model_param_on_cpu), 'param count not match'
            # for idx, tensor in self.model_param_on_gpu.items():
            #     assert tensor.is_cuda, 'model_param_on_gpu should be on GPU'
            #     t_cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
            #     t_cpu.data.copy_(tensor.data, non_blocking=True)
            #     self.model_param_on_cpu[id(tensor)] = t_cpu

            for idx, tensor in self.model_param_on_gpu.items():
                # 1. 安全获取预分配的 CPU buffer
                if idx not in self.model_param_on_cpu:
                    print(f"Error: {idx} not in model_param_on_cpu")
                    continue
                t_cpu = self.model_param_on_cpu[idx]

                # 2. 这里的 tensor 是 GPU Parameter
                # 直接拷贝，利用已经分配好的 t_cpu 内存
                t_cpu.copy_(tensor.data, non_blocking=True)

            if async_offload:
                # record offload event
                self.offload_event.record(stream)
            else:
                # wait for all tensors offloaded
                self.compute_stream.wait_stream(stream)
                self.offload_timestamp[1] = time.time()
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
        # self.check_model_param_device()
        return self.offload_timestamp

    def offload_finished(self):
        return self.offload_event.query()

    def reload_finished(self):
        return self.reload_event.query()

    def _release_gpu_memory(self):
        # Release model parameters and optimizer states from GPU
        # print(f'before release GPU memory, GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} MB')
        offload_byte = 0
        for idx in self.model_param_on_gpu.keys():
            if idx in self.except_tensor_idx_list:
                continue
            offload_byte += self.model_param_on_gpu[idx].numel() * self.model_param_on_gpu[idx].element_size()
            # self.model_param_on_gpu[idx].data = torch.empty(0, device=self.device)
            self.model_param_on_gpu[idx].data = torch.empty(0, device='cpu')
        # torch.cuda.empty_cache()
        # print(
        #     f'after release GPU memory, GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} MB,total M bytes offloaded: {offload_byte/1024/1024}'
        # )

    # reload model parameters and optimizer states from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else self.compute_stream
        stream.wait_stream(self.compute_stream)  # reload should be done after compute
        # used for profiling
        self.start_reload_event.record(stream)
        self.start_reload_event.synchronize()
        self.reload_timestamp[0] = time.time()
        # ----------------------------------------------------
        with torch.cuda.stream(stream):
            # Reload model parameters from CPU to GPU
            for idx, tensor in self.model_param_on_cpu.items():
                assert tensor.is_cpu, 'model_param_on_cpu should be on CPU DRAM'
                t_gpu = torch.empty_like(tensor, device=self.device)  # no pin_memory on GPU
                t_gpu.data.copy_(tensor.data, non_blocking=True)
                self.model_param_on_gpu[idx].data = t_gpu.data
            if async_reload:
                # record reload event
                self.reload_event.record(stream)
            else:
                # wait for all tensors reloaded
                self.compute_stream.wait_stream(stream)
                self.reload_timestamp[1] = time.time()

    def wait_reload(self):
        if self.load_stream != self.compute_stream:
            self.compute_stream.wait_event(self.reload_event)
            # used for profiling
            self.reload_event.synchronize()
            elapsed_time = self.start_reload_event.elapsed_time(self.reload_event)  # kernel time in ms
            self.reload_timestamp[1] = self.reload_timestamp[0] + elapsed_time / 1000  # time in seconds
        return self.reload_timestamp

    def check_model_param_device(self):
        print("--- Checking Parameter Devices ---")
        cuda_count = 0
        cpu_count = 0
        for name, param in self.base_model.named_parameters():
            if param.device.type == 'cuda':
                cuda_count += 1
            else:
                cpu_count += 1
            # 打印前几个看看
            # if cuda_count < 100 and param.device.type == 'cuda':
            #     print(f"CUDA: {name}")
        print(f"Total: {cuda_count} on CUDA, {cpu_count} on CPU")
