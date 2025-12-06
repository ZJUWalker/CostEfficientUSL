import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from typing import Any, List, Dict, Set, Union, Tuple
from collections import defaultdict
import time

import torch
import torch.nn as nn
from .activation_offload import CpuOffloadSavedTensorHook


class ParamState:
    def __init__(self, param: torch.nn.Parameter) -> None:
        # 始终持有原始 Parameter 对象
        self.param = param

        # 记录元数据：(size, dtype, stride, storage_offset)
        # 注意：第一次初始化时，这里的 param 就是原始参数本身
        self.tensor_meta = [(param.size(), param.dtype, param.stride(), param.storage_offset())]
        self.ref_cnt = 1
        self.device = param.device
        self.offloaded = False
        self.reloaded = False

        self.cpu_backup = None
        self.prefetch_buffer = None

    def add_tensor(self, tensor: torch.Tensor) -> None:
        # 校验：确保指向同一块内存
        assert tensor.data_ptr() == self.param.data_ptr()
        self.ref_cnt += 1
        # 【关键】记录具体这个 View 的几何特征
        self.tensor_meta.append((tensor.size(), tensor.dtype, tensor.stride(), tensor.storage_offset()))

    def get_tensor(self) -> torch.Tensor:
        return self.param

    def get_ref_cnt(self) -> int:
        return self.ref_cnt

    def get_reloaded_tensor(self, ref_cnt) -> torch.Tensor:
        self.ref_cnt -= 1
        assert self.ref_cnt >= 0

        # 取出当初存的元数据
        target_size, target_dtype, target_stride, target_offset = self.tensor_meta[ref_cnt - 1]

        # 【核心修复】
        # 不要用 self.prefetch_buffer.view()，因为 view() 处理不了转置(Transpose)。
        # 我们直接基于恢复后的 self.param 使用 as_strided 重建视图。
        # self.param 此时已经通过 reload() 恢复了显存数据。

        return self.param.as_strided(size=target_size, stride=target_stride, storage_offset=target_offset)

    def offload(self, pin_memory=True) -> None:
        assert not self.offloaded

        # 【关键】永远只备份原始参数 (self.param)
        # 这样能保证备份的数据布局是 Canonical 的 [6144, 2048]
        self.cpu_backup = torch.empty(self.param.size(), dtype=self.param.dtype, layout=self.param.layout, device="cpu", pin_memory=pin_memory)
        self.cpu_backup.copy_(self.param, non_blocking=pin_memory)

        # 释放显存：把原始参数指向空
        self.param.data = torch.empty(0, dtype=self.param.dtype, device=self.param.device)
        self.offloaded = True

    def create_prefetch_buffer(self) -> None:
        if self.prefetch_buffer is None:
            # Buffer 大小应该等于原始参数大小
            self.prefetch_buffer = torch.empty(self.cpu_backup.size(), dtype=self.cpu_backup.dtype, layout=self.cpu_backup.layout, device=self.device)

    def reload(self) -> None:
        assert not self.reloaded and self.offloaded
        # 1. GPU 上的 Buffer 接收 CPU 数据
        self.prefetch_buffer.copy_(self.cpu_backup, non_blocking=True)

        # 2. 【关键】恢复原始参数的数据
        self.param.data = self.prefetch_buffer

        self.reloaded = True


"""
 this is a hook to offload model parameters to device when model params are captured by autograd.
"""


class AsyncModelParamOffloadHandler(CpuOffloadSavedTensorHook):

    def __init__(
        self,
        model: nn.Module,
        # num_minibatch,  # must be <= actual batchnumber of groups (number of commits)
        device='cuda',
        load_stream: torch.cuda.Stream = None,
        offload_stream: torch.cuda.Stream = None,
    ) -> None:
        self.model = model
        # self.num_minibatch = num_minibatch
        self.device = device
        self.load_stream = load_stream if load_stream else torch.cuda.Stream(torch.cuda.current_device())
        self.offload_stream = offload_stream if offload_stream else torch.cuda.Stream(torch.cuda.current_device())
        self.compute_stream = torch.cuda.current_stream(self.device)

        self.start_offload_event = torch.cuda.Event(enable_timing=True)
        self.start_reload_event = torch.cuda.Event(enable_timing=True)
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)
        self.offload_timestamp = [0, 0]
        self.reload_timestamp = [0, 0]
        self.offload_tensor_type = 0  # offloading flag
        self.torch_tensor_count = 0
        self.total_offload_size = 0
        self.tensor_tag_to_state: Dict[Tuple[int, int], ParamState] = {}
        self.offloaded_tensor_buffers = []
        self._init_model_info()
        pass

    def _tensor_need_offloading_checker(self, tensor: torch.Tensor):
        return tensor.data_ptr() in self.ptr_to_param.keys()

    def _init_model_info(self):
        # self.ptr_to_param: List[int] = []
        self.param_name_ptr_dict: Dict[int, str] = {p.data_ptr(): n for n, p in self.model.named_parameters()}
        self.ptr_to_param = {p.data_ptr(): p for p in self.model.parameters()}

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        if self._tensor_need_offloading_checker(tensor):
            # print(
            #     f"pushing tensor {tensor.shape}(param shape:{self.ptr_to_param[tensor.data_ptr()].shape}) to device {self.device},is leaf {tensor.is_leaf},name : {self.param_name_ptr_dict[tensor.data_ptr()]}"
            # )
            original_param = self.ptr_to_param[tensor.data_ptr()]
            tensor_tag = (self.offload_tensor_type, tensor.data_ptr())
            if tensor_tag not in self.tensor_tag_to_state:
                self.tensor_tag_to_state[tensor_tag] = ParamState(original_param)
                # 简单做法：
                state = self.tensor_tag_to_state[tensor_tag]
                # 覆盖掉 init 里默认记录的 param meta，改为记录当前 tensor 的 meta
                state.tensor_meta[0] = (tensor.size(), tensor.dtype, tensor.stride(), tensor.storage_offset())
            else:
                self.tensor_tag_to_state[tensor_tag].add_tensor(tensor)
            tensor_tag = (tensor_tag, self.tensor_tag_to_state[tensor_tag].get_ref_cnt())
        else:
            # print(f"stash tensor {tensor.shape} to device {self.device},is not in target_storage_ptrs")
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag: Tuple[int, int], **kwargs):
        """Tensor pop."""
        if isinstance(tensor_tag[0], tuple):
            tensor_tag, ref_cnt = tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor_or_state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(tensor_or_state, ParamState):
            tensor = tensor_or_state.get_reloaded_tensor(ref_cnt)
            if tensor_or_state.get_ref_cnt() > 0:
                self.tensor_tag_to_state[tensor_tag] = tensor_or_state
        else:
            tensor = tensor_or_state
        assert not isinstance(tensor, ParamState)
        return tensor
        pass

    def offload(self, async_off: bool = False):
        # the copying of this minibatch should wait for the computation stream
        self.offload_stream.wait_stream(self.compute_stream)
        self.offload_stream.record_event(self.start_offload_event)
        self.start_offload_event.synchronize()
        self.offload_timestamp[0] = time.time()
        # print('record d2h start event for mb_idx', mb_to_offload)
        # print(f'model offload tensor state count: {len(self.tensor_tag_to_state)}')
        with torch.cuda.stream(self.offload_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                tensor_type, _ = tensor_tag
                if tensor_type == self.offload_tensor_type and isinstance(state, ParamState):
                    tensor_on_device = state.get_tensor()
                    assert self._tensor_need_offloading_checker(tensor_on_device)
                    # if offload, return the reference to cpu copy
                    if tensor_on_device is not None:
                        self.total_offload_size += tensor_on_device.numel() * tensor_on_device.element_size()
                        # print(f"offloading tensor size: {self.total_offload_size / (1024.0**3):.5f} GiB")
                        state.offload()
                        # save the tensor since this the copy of this tensor has not yet finished
                        self.offloaded_tensor_buffers.append(tensor_on_device)
        self.offload_stream.record_event(self.offload_event)
        if not async_off:
            # self.offload_stream.wait_event(self.offload_event)
            self.compute_stream.wait_event(self.offload_event)
            self.offload_event.synchronize()
            self.offload_timestamp[1] = time.time()
            self.offloaded_tensor_buffers.clear()
        # print(f"offloading tensor size: {self.total_offload_size / (1024.0**3):.5f} GiB")

    def wait_offload(self):
        self.compute_stream.wait_event(self.offload_event)
        self.offload_event.synchronize()
        offload_time = self.start_offload_event.elapsed_time(self.offload_event)
        # end_offload = time.time()
        self.offloaded_tensor_buffers.clear()  # release the memory of offloaded tensors
        self.offload_timestamp[1] = self.offload_timestamp[0] + offload_time / 1000
        return self.offload_timestamp

    def reload(self, async_reload=False):
        self.load_stream.wait_stream(self.compute_stream)
        self.load_stream.record_event(self.start_reload_event)
        self.start_reload_event.synchronize()
        self.reload_timestamp[0] = time.time()
        if len(self.tensor_tag_to_state) == 0:
            self.model.to(self.device)  # the fisrt time of reload, we need to move the model to device
            return
        for tensor_label, state in self.tensor_tag_to_state.items():
            tensor_type, _ = tensor_label
            if tensor_type == self.offload_tensor_type:
                if isinstance(state, ParamState):
                    state.create_prefetch_buffer()
        with torch.cuda.stream(self.load_stream):
            # move back tensors
            for tensor_label, state in self.tensor_tag_to_state.items():
                tensor_type, _ = tensor_label
                if tensor_type == self.offload_tensor_type:
                    if isinstance(state, ParamState):
                        state.reload()
        self.load_stream.record_event(self.reload_event)
        if not async_reload:
            # self.load_stream.wait_event(self.reload_event)
            self.compute_stream.wait_event(self.reload_event)
            self.reload_event.synchronize()
            self.reload_timestamp[1] = time.time()
        # print(f"reloading tensor size: {self.total_offload_size / (1024.0**3):.5f} GiB")

    def wait_reload(self):
        self.compute_stream.wait_event(self.reload_event)
        self.reload_event.synchronize()
        # end_reload = time.time()
        reload_time = self.start_reload_event.elapsed_time(self.reload_event)
        self.reload_timestamp[1] = self.reload_timestamp[0] + reload_time / 1000
        return self.reload_timestamp

    def on_save_for_backward(self, tensor: torch.Tensor):
        return self.tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any):
        return self.tensor_pop(saved_state)

    def update_param_ptr(self):
        self._init_model_info()
