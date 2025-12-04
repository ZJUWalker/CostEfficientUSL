import torch
from usl.offload import CpuOffloadSavedTensorHook, AsyncModelParamOffloadHandler, AsyncDoubleBufferGroupOffloadHandler
from typing import Any, List, Tuple, Union

"""
HybridOffloadContext 用于在模型训练过程中，将模型参数和激活值分开进行卸载。
"""


class HybridOffloadContext(CpuOffloadSavedTensorHook):
    def __init__(self, model_handler: AsyncModelParamOffloadHandler, activation_handler: AsyncDoubleBufferGroupOffloadHandler, debug=False) -> None:
        super().__init__()
        self.model_param_handler = model_handler
        self.activation_handler = activation_handler
        self.debug = debug

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        # 1. 优先判断是否为模型参数
        if self.model_param_handler._tensor_need_offloading_checker(tensor):
            # 交给模型卸载器处理
            tag = self.model_param_handler.tensor_push(tensor)
            # 返回带前缀的 tag，用于区分归属
            return ("model", tag)

        # 2. 如果不是参数，则尝试交给激活值卸载器
        # 注意：这里我们不需要再检查 activation_handler 的 checker，
        # 因为我们希望所有非参数的 Tensor 尽可能进入激活卸载流程（或者由它决定是否 stash）
        else:
            tag = self.activation_handler.tensor_push(tensor)
            return ("act", tag)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        # 解析 Tag
        owner, tag = saved_state

        if owner == "model":
            return self.model_param_handler.tensor_pop(tag)
        elif owner == "act":
            return self.activation_handler.tensor_pop(tag)
        else:
            raise ValueError(f"Unknown tensor owner: {owner}")
