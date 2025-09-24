import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union


class ActivationCollector:

    def __init__(self, base_model: nn.Module, offload_threshold: int = 1024, device=None):
        self.base_model = base_model
        self.activations_on_gpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> activation on gpu
        self.activations_on_cpu: Dict[int, torch.Tensor] = {}  # tensor_ptr -> activation on cpu DRAM
        self.offload_threshold = offload_threshold  # Byte
        self.device = device
        self._register_fwd_hook()

    def _register_fwd_hook(self):
        self.hooks = []

        def after_fwd_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Union[torch.Tensor, Tuple[torch.Tensor]]):
            if isinstance(output, torch.Tensor):
                if output.numel() * output.element_size() > self.offload_threshold:
                    tensor_ptr = output.data_ptr()
                    if tensor_ptr not in self.activations_on_gpu:
                        self.activations_on_gpu[tensor_ptr] = output
                    else:
                        print(f"Activation {tensor_ptr} already on GPU")
            elif isinstance(output, List) or isinstance(output, Tuple) or isinstance(output, Dict):
                for tensor in output:
                    if isinstance(tensor, torch.Tensor):
                        if tensor.numel() * tensor.element_size() > self.offload_threshold:
                            tensor_ptr = tensor.data_ptr()
                            if tensor_ptr not in self.activations_on_gpu:
                                self.activations_on_gpu[tensor_ptr] = tensor
                            else:
                                print(f"Activation {tensor_ptr} already on GPU")
            pass

        # TODO dynamically release memory of tensors in activations_on_gpu when they are no longer needed during backward phase
        def after_bwd_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            print(f"Module backward hook called")
            pass

        for name, module in self.base_model.named_modules():
            hk = module.register_forward_hook(after_fwd_hook)
            self.hooks.append(hk)

    # offload activations from GPU to CPU,except for tensors in except_tensor_idx_list
    def offload(self, except_tensor_idx_list: List[str] = []):

        for idx, tensor in self.activations_on_gpu.items():
            if self.device is None:
                self.device = tensor.device
            if idx in except_tensor_idx_list:
                continue
            t_cpu = torch.empty_like(tensor, device='cpu', pin_memory=True)
            t_cpu.data.copy_(tensor.data, non_blocking=True)
            self.activations_on_cpu[idx] = t_cpu
        torch.cuda.synchronize(self.device)
        # release GPU memory
        for idx in self.activations_on_gpu.keys():
            if idx in except_tensor_idx_list:
                continue
            self.activations_on_gpu[idx].data = torch.empty(0)
        torch.cuda.empty_cache()

    # reload activations from CPU to GPU
    def reload(self):
        for idx, tensor in self.activations_on_cpu.items():
            t_gpu = torch.empty_like(tensor, device=self.device)  # no pin_memory on GPU
            t_gpu.data.copy_(tensor.data, non_blocking=True)
            self.activations_on_gpu[idx].data = t_gpu.data
        torch.cuda.synchronize(self.device)
        # release CPU memory
        self.activations_on_cpu.clear()
        torch.cuda.empty_cache()

    # clear all activations on CPU and GPU,called after batch bwd
    def clear(self):
        self.activations_on_cpu.clear()
        self.activations_on_gpu.clear()
        torch.cuda.empty_cache()
