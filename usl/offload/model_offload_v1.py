import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from typing import List, Dict, Tuple, Set, Union
import time


class ModelParamOffloadHook:

    def __init__(
        self,
        base_model: nn.Module,
        offload_threshold: int = 0,
        offload_layer_num: int = 1000,  # 默认全部卸载
        device="cuda",
        load_stream=None,
        offload_stream=None,
        except_tensor_idx_list=[],
    ):
        self.base_model = base_model
        self.offload_threshold = offload_threshold
        self.offload_layer_num = offload_layer_num
        self.device = device

        # Stream 管理 (保持你原有的逻辑)
        self.load_stream: torch.cuda.Stream = load_stream if load_stream else torch.cuda.Stream(device)
        self.offload_stream: torch.cuda.Stream = offload_stream if offload_stream else torch.cuda.Stream(device)
        self.compute_stream: torch.cuda.Stream = torch.cuda.current_stream(self.device)

        # Events (保持用于同步和计时)
        self.start_offload_event = torch.cuda.Event(enable_timing=True)
        self.start_reload_event = torch.cuda.Event(enable_timing=True)
        self.offload_event = torch.cuda.Event(enable_timing=True)
        self.reload_event = torch.cuda.Event(enable_timing=True)
        self.offload_timestamp = [0, 0]
        self.reload_timestamp = [0, 0]

        self.except_tensor_idx_list: List[int] = except_tensor_idx_list

        # === 新增/修改的核心数据结构 ===
        # 1. 候选名单：在 Init 时确定哪些参数有资格被卸载 (根据 layer_num 和 threshold)
        self.target_param_ptrs: Set[int] = set()
        self._identify_target_params()

        # 2. 动态捕获列表：Forward 过程中 Hook 抓到的参数
        self.captured_params: List[torch.Tensor] = []

        # 3. CPU 缓冲区：offload 后存储参数的地方 {index_in_captured_list -> cpu_tensor}
        self.cpu_buffers: Dict[int, torch.Tensor] = {}

        self.is_offloaded = False
        self.hook_handle = None

    def _identify_target_params(self):
        """
        初始化阶段：根据层数和大小阈值，圈定哪些参数应该被 Hook 捕获。
        这替代了你原来的 _init_param_dict
        """
        curr_layer_idx = 0
        count = 0

        for name, param in self.base_model.named_parameters():
            # 1. 排除在黑名单里的
            if param.data_ptr() in self.except_tensor_idx_list:
                continue

            # 2. 排除太小的
            if param.numel() * param.element_size() < self.offload_threshold:
                continue

            # 3. 排除超过层数限制的
            if self.offload_layer_num > 0:
                # 简单的层数解析逻辑，沿用你的写法
                if f'h.{curr_layer_idx}' in name or f'layers.{curr_layer_idx}' in name:
                    curr_layer_idx += 1

                if curr_layer_idx > self.offload_layer_num:
                    continue

            # 符合条件，加入“白名单”
            # 注意：这里使用 data_ptr() 也就是显存物理地址作为唯一标识，比 id() 更安全
            self.target_param_ptrs.add(param.data_ptr())
            count += 1

        print(f"[Init] Identified {count} parameters eligible for offloading.")

    # ================= Hook Logic =================

    def pack_hook(self, tensor: torch.Tensor) -> Union[int, torch.Tensor]:
        # 核心逻辑：只有在 target_param_ptrs 白名单里的参数才处理
        if tensor.is_cuda and tensor.data_ptr() in self.target_param_ptrs:
            # 1. 记录到列表，生成 ID
            token_id = len(self.captured_params)
            self.captured_params.append(tensor)

            # 2. 返回 ID 给 Autograd，切断 Autograd 对 GPU 显存的直接引用
            return token_id

        return tensor

    def unpack_hook(self, token: Union[int, torch.Tensor]):
        # 如果是我们的 token ID
        if isinstance(token, int):
            idx = token
            original_tensor = self.captured_params[idx]

            # 如果发生了 offload，需要从 CPU 临时搬回 GPU
            if self.is_offloaded:
                # 注意：这里我们利用 load_stream 进行传输，但必须立刻同步，
                # 因为 Autograd 马上就要拿这个 tensor 去算梯度了，不能延迟。
                # 为了性能，这里通常只能用 non_blocking=True，依靠 PyTorch 自身的流管理
                if idx in self.cpu_buffers:
                    return self.cpu_buffers[idx].cuda(non_blocking=True)
                else:
                    # 容错：如果找不到 buffer，只能返回原始的（可能已经是 CPU 了）
                    return original_tensor

            # 如果没 offload，直接返回原始 tensor
            return original_tensor

        return token

    # ================= Context Manager =================
    def __enter__(self):
        self.hook_handle = saved_tensors_hooks(self.pack_hook, self.unpack_hook)
        self.hook_handle.__enter__()
        return self

    def __exit__(self, *args):
        self.hook_handle.__exit__(*args)

    # ================= Manual Operations (Optimized) =================

    def offload(self, async_offload=False):
        """
        Forward 结束后调用。将 Hook 捕获的所有参数搬运到 CPU。
        【优化点】：自动去重，处理共享权重 (Weight Tying)
        """
        if self.is_offloaded:
            return

        stream = self.offload_stream
        stream.wait_stream(self.compute_stream)

        self.start_offload_event.record(stream)
        self.offload_timestamp[0] = time.time()

        # 临时字典，用于记录本次 Offload 已经搬运过的存储块
        # Key: 原始 GPU Tensor 的 data_ptr
        # Value: 对应的 CPU Tensor
        gpu_to_cpu_map: Dict[int, torch.Tensor] = {}

        with torch.cuda.stream(stream):
            for idx, tensor in enumerate(self.captured_params):
                # 0. 边缘情况检查：如果列表里有同一个 Tensor 对象出现两次
                # 第二次遍历时它已经是 CPU Tensor 了，直接跳过拷贝，但要记录 buffer
                if not tensor.is_cuda:
                    # 此时 tensor 已经是 CPU 的了，它本身就是 buffer
                    # 注意：这种情况极少见，除非 Hook 捕获了同一个 Parameter 对象两次
                    self.cpu_buffers[idx] = tensor
                    continue

                # 1. 获取物理地址 ID
                gpu_ptr = tensor.data_ptr()

                if gpu_ptr not in gpu_to_cpu_map:
                    # === Case A: 第一次遇到这块显存 ===
                    # 创建 CPU 副本并拷贝
                    cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    cpu_tensor.copy_(tensor, non_blocking=True)

                    # 记录到去重字典
                    gpu_to_cpu_map[gpu_ptr] = cpu_tensor
                else:
                    # === Case B: 这块显存之前搬过了 (共享权重) ===
                    # 直接复用已有的 CPU Tensor，不再消耗带宽和内存
                    cpu_tensor = gpu_to_cpu_map[gpu_ptr]

                # 2. 记录到 Buffer 列表 (供 unpack_hook 使用)
                # 即使是重复的参数，索引 idx 是不同的，所以都要记录
                self.cpu_buffers[idx] = cpu_tensor

                # 3. 【核心】指针置换
                # 多个共享的 Parameter 都会指向同一个 cpu_tensor 的 data
                tensor.data = cpu_tensor.data

        if async_offload:
            self.offload_event.record(stream)
        else:
            stream.synchronize()
            self.offload_timestamp[1] = time.time()
            # self._release_gpu_memory_check()

        self.is_offloaded = True

    def reload(self, async_reload=False):
        """
        Optimizer Step 之前调用。将参数搬回 GPU。
        【优化点】：去重上传，保证共享权重关系恢复
        """
        if not self.is_offloaded:
            return

        stream = self.load_stream
        stream.wait_stream(self.compute_stream)

        self.start_reload_event.record(stream)
        self.reload_timestamp[0] = time.time()

        # 临时字典，用于记录本次 Reload 已经搬回的存储块
        # Key: CPU Tensor 的 data_ptr
        # Value: 新分配的 GPU Tensor
        cpu_to_gpu_map: Dict[int, torch.Tensor] = {}

        with torch.cuda.stream(stream):
            for idx, tensor in enumerate(self.captured_params):
                # 从 buffer 里拿到 CPU 数据
                if idx in self.cpu_buffers:
                    cpu_tensor = self.cpu_buffers[idx]
                    cpu_ptr = cpu_tensor.data_ptr()

                    if cpu_ptr not in cpu_to_gpu_map:
                        # === Case A: 第一次搬回这块数据 ===
                        gpu_tensor = cpu_tensor.cuda(non_blocking=True)
                        cpu_to_gpu_map[cpu_ptr] = gpu_tensor
                    else:
                        # === Case B: 之前搬过了 ===
                        gpu_tensor = cpu_to_gpu_map[cpu_ptr]

                    # 恢复 Parameter 指向 (共享关系的 Parameter 会指向同一块 GPU 显存)
                    tensor.data = gpu_tensor.data

        if async_reload:
            self.reload_event.record(stream)
        else:
            stream.synchronize()
            self.reload_timestamp[1] = time.time()

            self.cpu_buffers.clear()
            self.is_offloaded = False

    def clear_buffer(self):
        """
        每个 Batch 结束后必须调用！清空 Hook 捕获列表。
        """
        self.captured_params.clear()
        self.cpu_buffers.clear()
        self.is_offloaded = False

    def _release_gpu_memory_check(self):
        # 辅助函数，仅用于打印验证
        print(f'GPU memory check after offload: {torch.cuda.memory_allocated()/1024**2:.2f} MB')
        # torch.cuda.empty_cache()

    # 保持原有的 wait 接口以便兼容你的测试代码
    def wait_offload(self):
        self.offload_event.synchronize()
        elapsed_time = self.start_offload_event.elapsed_time(self.offload_event)  # kernel time in ms
        self.offload_timestamp[1] = self.offload_timestamp[0] + elapsed_time / 1000  # time in seconds
        # self._release_gpu_memory_check()
        return self.offload_timestamp

    def wait_reload(self):
        self.reload_event.synchronize()
        self.cpu_buffers.clear()  # reload 完确认清理
        self.is_offloaded = False
        elapsed_time = self.start_reload_event.elapsed_time(self.reload_event)  # kernel time in ms
        self.reload_timestamp[1] = self.reload_timestamp[0] + elapsed_time / 1000  # time in seconds
        return self.reload_timestamp
