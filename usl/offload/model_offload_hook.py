import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from typing import Any, List, Dict, Optional, Set, Union, Tuple
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



class LayerwiseAsyncModelParamOffloadHandler(AsyncModelParamOffloadHandler):
    """
    基于层范围的异步参数卸载 Handler，继承自 AsyncModelParamOffloadHandler。
    
    核心改进：
    - 支持通过参数名精确识别模型层数（Llama/Qwen/GPT 等结构）
    - 仅卸载指定层范围 [offload_start_layer, offload_end_layer) 内的参数
    - 支持参数大小阈值过滤（小参数保留 GPU，避免通信开销）
    - 完全复用父类的 ParamState 管理、异步流控制和 saved_tensors_hooks 机制
    
    适用场景：
    - Pipeline Parallelism 中仅卸载当前 stage 的后几层
    - 大模型微调时仅卸载冻结层（frozen layers）
    - 与 FSDP2 结合时，仅卸载特定 TP 组内的层
    """
    
    def __init__(
        self,
        model: nn.Module,
        device='cuda',
        load_stream: torch.cuda.Stream = None,
        offload_stream: torch.cuda.Stream = None,
        offload_layer_num:int=0,
        offload_start_layer: int = 0,
        offload_end_layer: Optional[int] = None,
        offload_threshold: int = 0,  # bytes, 小于此阈值的参数保留在 GPU
    ) -> None:
        # 必须先保存层配置，因为父类 __init__ 会调用 _init_model_info
        self.offload_start_layer = offload_start_layer
        self.offload_end_layer = offload_end_layer if offload_end_layer is not None else offload_start_layer+offload_layer_num
        self.offload_threshold = offload_threshold
        
        # 层统计信息
        self.ptr_to_layer_idx: Dict[int, int] = {}  # data_ptr -> layer_idx
        self.offload_target_ptrs: Set[int] = set()  # 被选中卸载的参数指针集合
        self.layer_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"param_count": 0, "total_bytes": 0, "offloaded": False})
        
        # 调用父类 __init__，它会触发 _init_model_info
        super().__init__(model, device, load_stream, offload_stream)
        
        # 父类初始化完成后，建立层映射并筛选目标参数
        self._setup_layerwise_offload()
    
    def _is_layer_root_module(self, module_name: str) -> bool:
        """
        判断模块名是否对应 Transformer 层根模块（如 model.layers.0）。
        支持标准命名：model.layers, transformer.h, model.decoder.layers 等。
        """
        if module_name in ["", "model", "transformer", "model.transformer"]:
            return False
        
        layer_prefix = [
            "model.layers",
            "model.transformer.layers",
            "transformer.layers",
            "transformer.h",
            "model.decoder.layers",
        ]
        for prefix in layer_prefix:
            if module_name.startswith(prefix):
                last_part = module_name.split(".")[-1]
                if last_part.isdigit():
                    return True
        return False
    
    def _get_layer_idx_from_param_name(self, param_name: str) -> int:
        """
        从参数全名提取层索引。返回 -1 表示非层参数（embedding, head, norm 等）。
        e.g., "model.layers.0.self_attn.q_proj.weight" -> 0
              "model.embed_tokens.weight" -> -1
        """
        layer_prefix = [
            "model.layers",
            "model.transformer.layers",
            "transformer.layers",
            "transformer.h",
            "model.decoder.layers",
        ]
        for prefix in layer_prefix:
            if prefix in param_name:
                try:
                    suffix = param_name.split(prefix + ".", 1)[1]
                    layer_idx_str = suffix.split(".")[0]
                    if layer_idx_str.isdigit():
                        return int(layer_idx_str)
                except (IndexError, ValueError):
                    continue
        return -1
    
    def _init_model_info(self):
        """
        重写以扩展父类的参数指针映射。父类建立 ptr_to_param 和 param_name_ptr_dict，
        我们在 _setup_layerwise_offload 中补充层信息。
        """
        # 调用父类实现，建立基础的 ptr_to_param 映射
        super()._init_model_info()
        # 注意：此时 self.model 已可用，self.ptr_to_param 已建立
    
    def _setup_layerwise_offload(self):
        """
        建立层到参数的映射，并根据层范围筛选需要卸载的参数指针。
        仅在 __init__ 中调用一次。
        """
        total_offload_size = 0
        total_offload_count = 0
        
        for name, param in self.model.named_parameters():
            ptr = param.data_ptr()
            layer_idx = self._get_layer_idx_from_param_name(name)
            param_bytes = param.numel() * param.element_size()
            
            self.ptr_to_layer_idx[ptr] = layer_idx
            self.layer_stats[layer_idx]["param_count"] += 1
            self.layer_stats[layer_idx]["total_bytes"] += param_bytes
            
            # 决策：是否将该参数加入卸载目标集合
            should_offload = True
            
            # 1. 层范围检查
            if layer_idx == -1:
                should_offload = False  # 非层参数（embedding, head, norm）保留
            elif not (self.offload_start_layer <= layer_idx < self.offload_end_layer):
                should_offload = False
            
            # 2. 大小阈值检查（避免卸载过小参数，通信开销 > 显存收益）
            if param_bytes < self.offload_threshold:
                should_offload = False
            
            if should_offload:
                self.offload_target_ptrs.add(ptr)
                self.layer_stats[layer_idx]["offloaded"] = True
                total_offload_size += param_bytes
                total_offload_count += param.numel()
        
        print(
            f"[LayerwiseAsyncOffload] Config: Layers [{self.offload_start_layer}, {self.offload_end_layer}), "
            f"Threshold {self.offload_threshold/1024**2:.2f} MB"
        )
        print(
            f"[LayerwiseAsyncOffload] Target: {len(self.offload_target_ptrs)} params, "
            f"{total_offload_count/1e6:.2f}M elements, "
            f"{total_offload_size/1024**3:.2f} GB"
        )
    
    def _tensor_need_offloading_checker(self, tensor: torch.Tensor) -> bool:
        """
        重写父类的卸载检查逻辑。只有满足以下条件的张量才会被 offload：
        1. 属于模型参数（继承父类检查）
        2. 在指定的层范围内
        3. 大于大小阈值
        """
        ptr = tensor.data_ptr()
        
        # 快速路径：不在目标集合中直接返回 False
        if ptr not in self.offload_target_ptrs:
            return False
        
        # 调用父类检查（确认是模型参数且满足其他内部条件）
        return super()._tensor_need_offloading_checker(tensor)
    
    def get_layer_offload_summary(self) -> Dict[int, Dict[str, Union[int, bool]]]:
        """
        获取每层参数的统计信息，用于调试层范围设置是否正确。
        
        Returns:
            Dict[int, Dict]: 层索引 -> {param_count, total_bytes(MB), offloaded(bool)}
        """
        summary = {}
        for layer_idx, stats in sorted(self.layer_stats.items()):
            summary[layer_idx] = {
                "param_count": stats["param_count"],
                "total_mb": stats["total_bytes"] / 1024**2,
                "offloaded": stats["offloaded"]
            }
        return summary
    
    def print_offload_layers(self, verbose: bool = False):
        """
        打印被卸载的层范围和非卸载层范围，用于快速验证配置。
        
        Args:
            verbose: 如果为 True，打印每层详细信息
        """
        offloaded_layers = []
        non_offloaded_layers = []
        
        for layer_idx in sorted(self.layer_stats.keys()):
            if self.layer_stats[layer_idx]["offloaded"]:
                offloaded_layers.append(layer_idx)
            else:
                non_offloaded_layers.append(layer_idx)
        
        # 压缩连续区间显示
        def compress_ranges(indices: List[int]) -> str:
            if not indices:
                return "None"
            ranges = []
            start = prev = indices[0]
            for curr in indices[1:] + [None]:
                if curr != prev + 1:
                    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                    start = curr
                prev = curr
            return ", ".join(ranges)
        
        # print(f"[LayerwiseAsyncOffload] Offloaded layers: {compress_ranges(offloaded_layers)}")
        # print(f"[LayerwiseAsyncOffload] GPU-resident layers: {compress_ranges(non_offloaded_layers)}")
        
        # if verbose:
        #     print("\nDetailed layer info:")
        #     for layer_idx in sorted(self.layer_stats.keys()):
        #         stats = self.layer_stats[layer_idx]
        #         status = "OFFLOAD" if stats["offloaded"] else "GPU"
        #         print(f"  Layer {layer_idx:3d}: {stats['param_count']:3d} params, "
        #               f"{stats['total_bytes']/1024**2:8.2f} MB [{status}]")
    
    def update_param_ptr(self):
        """
        重写父类方法：在模型参数指针变化时（如 after backward 清理），
        需要重建层映射。注意：这会重置 offload_target_ptrs，需谨慎调用。
        """
        # 先调用父类更新基础映射
        super().update_param_ptr()
        # 重建层映射
        self._setup_layerwise_offload()