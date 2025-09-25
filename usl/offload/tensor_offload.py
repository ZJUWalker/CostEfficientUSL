import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from .activation_offload import ActivationOffload
from .model_offload import ModelParamOffload
from .optimizer_offload import OptimizerStateOffload


class TensorOffload:

    def __init__(self, model, optimizer, offload_mp=True, offload_optim=True, offload_activ=True, offload_threshold: int = 0, device='cuda:0'):
        self.model = model
        self.optimizer = optimizer
        self.offload_threshold = offload_threshold
        self.device = device
        self.swap_stream = torch.cuda.Stream(device)
        self.activation_manager = (
            ActivationOffload(model, offload_threshold=offload_threshold, device=device, swap_stream=self.swap_stream) if offload_activ else None
        )
        self.model_param_manager = (
            ModelParamOffload(model, offload_threshold=offload_threshold, device=device, swap_stream=self.swap_stream) if offload_mp else None
        )
        self.optimizer_state_manager = (
            OptimizerStateOffload(optimizer, offload_threshold=offload_threshold, device=device, swap_stream=self.swap_stream)
            if offload_optim
            else None
        )

    def offload_model_param(self, async_offload=False):
        if self.model_param_manager is not None:
            self.model_param_manager.offload(async_offload=async_offload)
        pass

    def wait_offload_model(self):
        if self.model_param_manager is not None:
            self.model_param_manager.wait_offload()
        pass

    def offload_optimizer_state(self, async_offload=False):
        if self.optimizer_state_manager is not None:
            self.optimizer_state_manager.offload(async_offload=async_offload)
        pass

    def wait_offload_optimizer(self):
        if self.optimizer_state_manager is not None:
            self.optimizer_state_manager.wait_offload()
        pass

    def offload_activation(self, except_tensor_idx_list: List[int] = [], async_offload=False):
        if self.activation_manager is not None:
            self.activation_manager.offload(except_tensor_idx_list=except_tensor_idx_list, async_offload=async_offload)
        pass

    def wait_offload_activation(self, except_tensor_idx_list: List[int] = []):
        if self.activation_manager is not None:
            self.activation_manager.wait_offload(except_tensor_idx_list=except_tensor_idx_list)
        pass

    def reload_activation(self, except_tensor_idx_list: List[int] = [], async_reload=False):
        if self.activation_manager is not None:
            self.activation_manager.reload(except_tensor_idx_list=except_tensor_idx_list, async_reload=async_reload)
        pass

    def wait_reload_activation(self, except_tensor_idx_list: List[int] = []):
        if self.activation_manager is not None:
            self.activation_manager.wait_reload(except_tensor_idx_list=except_tensor_idx_list)
        pass

    def reload_model(self, async_reload=False):
        if self.model_param_manager is not None:
            self.model_param_manager.reload(async_reload=async_reload)
        pass
