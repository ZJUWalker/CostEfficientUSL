import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union


class ActivationOffload:

    def __init__(
        self, base_model: nn.Module, offload_threshold: int = 1024, device=None, load_stream=None, offload_stream=None
    ):

        self.base_model = base_model
        self.activations_on_gpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> activation on gpu
        self.activations_on_cpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> activation on cpu DRAM
        self.offload_threshold = offload_threshold  # Byte
        self.device = device
        self.load_stream = load_stream if load_stream is not None else torch.cuda.default_stream()
        self.offload_stream = offload_stream if offload_stream is not None else torch.cuda.default_stream()
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)
        self._register_fwd_hook()

    def _register_fwd_hook(self):
        self.hooks = []

        def after_fwd_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: Union[torch.Tensor, Tuple[torch.Tensor]]
        ):
            if isinstance(output, torch.Tensor):
                if output.numel() * output.element_size() >= self.offload_threshold and output.requires_grad:
                    tensor_ptr = output.data_ptr()
                    if tensor_ptr not in self.activations_on_gpu:
                        self.activations_on_gpu[tensor_ptr] = output
                    # else:
                    #     print(f"Activation {tensor_ptr},shape {output.shape},grad_fn {output.grad_fn} already on GPU")
            elif isinstance(output, List) or isinstance(output, Tuple) or isinstance(output, Dict):
                for tensor in output:
                    if isinstance(tensor, torch.Tensor):
                        if tensor.numel() * tensor.element_size() >= self.offload_threshold and tensor.requires_grad:
                            tensor_ptr = tensor.data_ptr()
                            if tensor_ptr not in self.activations_on_gpu:
                                self.activations_on_gpu[tensor_ptr] = tensor
                            # else:
                            #     print(
                            #         f"Activation {tensor_ptr},shape {tensor.shape},grad_fn {tensor.grad_fn} already on GPU"
                            #     )
            pass

        # TODO dynamically release memory of tensors in activations_on_gpu when they are no longer needed during backward phase
        def after_bwd_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            print(f"Module backward hook called")
            pass

        for name, module in self.base_model.named_modules():
            hk = module.register_forward_hook(after_fwd_hook)
            self.hooks.append(hk)

    # offload activations from GPU to CPU,except for tensors in except_tensor_idx_list
    def offload(self, except_tensor_idx_list: List[int] = [], async_offload=False):
        stream = self.offload_stream if self.offload_stream else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for idx, tensor in self.activations_on_gpu.items():
                if self.device is None:
                    self.device = tensor.device
                if idx in except_tensor_idx_list:
                    continue
                t_cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                t_cpu.data.copy_(tensor.data, non_blocking=True)
                self.activations_on_cpu[idx] = t_cpu
        if async_offload:
            # record offload event
            self.offload_event.record(stream)
        else:
            # wait for all tensor offloaded
            torch.cuda.synchronize(self.device)
            # release GPU memory
            self._release_tensor_on_gpu(except_tensor_idx_list)

    # wait for all activations offloaded to CPU,except for tensors in except_tensor_idx_list
    def wait_offload(self, except_tensor_idx_list: List[int] = []):
        if self.offload_stream != torch.cuda.current_stream():
            torch.cuda.current_stream().wait_event(self.offload_event)
            self._release_tensor_on_gpu(except_tensor_idx_list)

    def _release_tensor_on_gpu(self, except_tensor_idx_list: List[int] = []):
        for idx in self.activations_on_gpu.keys():
            if idx in except_tensor_idx_list:
                continue
            self.activations_on_gpu[idx].data = torch.empty(0)
        # torch.cuda.empty_cache()

    # reload activations from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for idx, tensor in self.activations_on_cpu.items():
                t_gpu = torch.empty_like(tensor, device=self.device)  # no pin_memory on GPU
                t_gpu.data.copy_(tensor.data, non_blocking=True)
                self.activations_on_gpu[idx].data = t_gpu.data
        if async_reload:
            # record reload event
            self.reload_event.record(stream)
        else:
            # wait for all tensor reloaded
            torch.cuda.synchronize(self.device)
            # release CPU memory
            self.activations_on_cpu.clear()
            # torch.cuda.empty_cache()

    def wait_reload(self):
        if self.load_stream != torch.cuda.current_stream():
            torch.cuda.current_stream().wait_event(self.reload_event)
            self.activations_on_cpu.clear()
            # torch.cuda.empty_cache()

    # clear all activations on CPU and GPU,called after batch bwd
    def clear(self):
        self.activations_on_cpu.clear()
        self.activations_on_gpu.clear()
        # torch.cuda.empty_cache()


"""
class ActivationOffloadAcrossBatches use to offload activations across batches, 

"""


class ActivationOffloadAcrossBatches:

    def __init__(
        self,
        base_model: nn.Module,
        micro_batch_num: int = 1,
        offload_threshold: int = 1024,
        device=None,
        load_stream=None,
        offload_stream=None,
    ):

        self.base_model = base_model
        self.curr_batch_idx = 0
        self.micro_batch_num = micro_batch_num
        self.activations_on_gpu: List[Dict[int, torch.Tensor]] = [{} for _ in range(micro_batch_num)]
        self.activations_on_cpu: List[Dict[int, torch.Tensor]] = [{} for _ in range(micro_batch_num)]
        self.offload_threshold = offload_threshold  # Byte
        self.device = device
        self.load_stream = load_stream if load_stream is not None else torch.cuda.default_stream()
        self.offload_stream = offload_stream if offload_stream is not None else torch.cuda.default_stream()
        self.offload_events = [torch.cuda.Event(enable_timing=True) for _ in range(micro_batch_num)]
        self.reload_events = [torch.cuda.Event(enable_timing=True) for _ in range(micro_batch_num)]
        self._register_fwd_hook()

    def _register_fwd_hook(self):
        self.hooks = []

        def after_fwd_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: Union[torch.Tensor, Tuple[torch.Tensor]]
        ):
            if isinstance(output, torch.Tensor):
                if output.numel() * output.element_size() >= self.offload_threshold:
                    tensor_ptr = output.data_ptr()
                    if tensor_ptr not in self.activations_on_gpu[self.curr_batch_idx]:
                        self.activations_on_gpu[self.curr_batch_idx][tensor_ptr] = output
                    else:
                        print(f"Activation {tensor_ptr} already on GPU")
            elif isinstance(output, List) or isinstance(output, Tuple) or isinstance(output, Dict):
                for tensor in output:
                    if isinstance(tensor, torch.Tensor):
                        if tensor.numel() * tensor.element_size() >= self.offload_threshold:
                            tensor_ptr = tensor.data_ptr()
                            if tensor_ptr not in self.activations_on_gpu[self.curr_batch_idx]:
                                self.activations_on_gpu[self.curr_batch_idx][tensor_ptr] = tensor
                            else:
                                print(f"Activation {tensor_ptr} already on GPU")
            pass

        # TODO dynamically release memory of tensors in activations_on_gpu when they are no longer needed during backward phase
        def after_bwd_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            print(f"Module backward hook called")
            pass

        # TODO
        def batch_end_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
            self.curr_batch_idx += 1
            if self.curr_batch_idx % self.micro_batch_num == 0:
                self.curr_batch_idx = 0

        for name, module in self.base_model.named_modules():
            hk = module.register_forward_hook(after_fwd_hook)
            self.hooks.append(hk)

        self.model_batch_end_hook = self.base_model.register_forward_hook(batch_end_hook)

    # offload activations from GPU to CPU,except for tensors in except_tensor_idx_list
    def offload(self, except_tensor_idx_list: List[int] = [], async_offload=False):
        stream = self.offload_stream if self.offload_stream else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for idx, tensor in self.activations_on_gpu.items():
                if self.device is None:
                    self.device = tensor.device
                if idx in except_tensor_idx_list:
                    continue
                t_cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                t_cpu.data.copy_(tensor.data, non_blocking=True)
                self.activations_on_cpu[idx] = t_cpu
        if async_offload:
            # record offload event
            self.offload_event.record(stream)
        else:
            # wait for all tensor offloaded
            torch.cuda.synchronize(self.device)
            # release GPU memory
            self._release_tensor_on_gpu(except_tensor_idx_list)

    # wait for all activations offloaded to CPU,except for tensors in except_tensor_idx_list
    def wait_offload(self, except_tensor_idx_list: List[int] = []):
        if self.offload_stream != torch.cuda.current_stream():
            torch.cuda.current_stream().wait_event(self.offload_event)
            self._release_tensor_on_gpu(except_tensor_idx_list)

    def _release_tensor_on_gpu(self, except_tensor_idx_list: List[int] = []):
        for idx in self.activations_on_gpu.keys():
            if idx in except_tensor_idx_list:
                continue
            self.activations_on_gpu[idx].data = torch.empty(0)
        torch.cuda.empty_cache()

    # reload activations from CPU to GPU
    def reload(self, async_reload=False):
        stream = self.load_stream if self.load_stream else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for idx, tensor in self.activations_on_cpu.items():
                t_gpu = torch.empty_like(tensor, device=self.device)  # no pin_memory on GPU
                t_gpu.data.copy_(tensor.data, non_blocking=True)
                self.activations_on_gpu[idx].data = t_gpu.data
        if async_reload:
            # record reload event
            self.reload_event.record(stream)
        else:
            # wait for all tensor reloaded
            torch.cuda.synchronize(self.device)
            # release CPU memory
            self.activations_on_cpu.clear()
            torch.cuda.empty_cache()

    def wait_reload(self):
        if self.load_stream != torch.cuda.current_stream():
            torch.cuda.current_stream().wait_event(self.reload_event)
            self.activations_on_cpu.clear()
            torch.cuda.empty_cache()

    # clear all activations on CPU and GPU,called after batch bwd
    def clear(self):
        self.activations_on_cpu.clear()
        self.activations_on_gpu.clear()
        torch.cuda.empty_cache()
