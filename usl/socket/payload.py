from dataclasses import dataclass, field
from typing import Any, Tuple, Literal, Dict, Optional, Union
import torch

@dataclass
class Payload:
    # 修改为 Any，因为在开启压缩时，它将存放 packed_indices (uint8 tensor)
    # 兼容原始的 16-bit 激活张量，以及开启 QLoRA 压缩后的字典 (含 packed_indices, pad, q_shape)
    tensor: Union[torch.Tensor, Dict[str, Any]]
    
    # 新增：用于存放反量化需要的元数据 (scales, min, max, original_shape等)
    aux: Optional[Dict[str, Any]] = None 
    
    # 新增：标志位，标识当前 Payload 是否经过了 QLoRA 压缩
    is_compressed: bool = False

    is_activation: bool = True
    phase: Literal["FWD", "BWD"] = "FWD"
    token: str = ""
    group_id: str = ""
    mb_idx: int = 0
    mb_total: int = 0
    attention_mask: Optional[torch.Tensor] = None
    position_embeddings: Optional[Tuple[torch.Tensor, ...]] = None

    def payload_nbytes(self) -> int:
        """计算 payload 中所有 tensor 的占用字节数（单位: Byte）"""
        total = 0

        def tensor_nbytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        try:
            # 1. 计算主 tensor (可能是原始激活，也可能是压缩后的 uint8 张量)
            if isinstance(self.tensor, torch.Tensor):
                total += tensor_nbytes(self.tensor)
            elif isinstance(self.tensor, dict):  # 兼容开启压缩后的 dict 结构
                for k, v in self.tensor.items():
                    if isinstance(v, torch.Tensor):
                        total += tensor_nbytes(v)
                
            # 2. 如果开启了压缩，累加 aux 字典中的额外张量开销 (如 scales_q)
            if self.is_compressed and self.aux is not None:
                for k, v in self.aux.items():
                    if isinstance(v, torch.Tensor):
                        total += tensor_nbytes(v)

            # 3. 计算其他控制信号
            for val in [self.attention_mask, self.position_embeddings]:
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    total += tensor_nbytes(val)
                elif isinstance(val, (list, tuple)):
                    for v in val:
                        if isinstance(v, torch.Tensor):
                            total += tensor_nbytes(v)
        except Exception as e:
            # 最好在这里加一行打印，防止静默吞掉异常
            print(f"[Warning] Failed to calculate payload bytes: {e}")
            return 0
        return total
    
    def to_cpu(self):
        """
        在网络发送前调用此方法，将 Payload 中所有张量剥离计算图并转移到 CPU，
        避免 Pickle 序列化带有 CUDA context 的张量引发错误。
        """
        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_cpu(v) for v in obj)
            return obj

        self.tensor = _to_cpu(self.tensor)
        if self.aux is not None:
            self.aux = _to_cpu(self.aux)
        self.attention_mask = _to_cpu(self.attention_mask)
        self.position_embeddings = _to_cpu(self.position_embeddings)
        
        return self
    
    def pin_memory(self):
        """
        在网络接收后调用此方法，将 Payload 中所有 CPU 张量锁页 (pin_memory)，
        从而加速后续向 GPU 的非阻塞拷贝。
        """
        def _pin(obj):
            if isinstance(obj, torch.Tensor):
                # 只有 CPU 上的 tensor 才能 pin_memory
                if not obj.is_cuda and not obj.is_pinned():
                    return obj.pin_memory()
                return obj
            elif isinstance(obj, dict):
                return {k: _pin(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_pin(v) for v in obj)
            return obj

        self.tensor = _pin(self.tensor)
        if self.aux is not None:
            self.aux = _pin(self.aux)
        self.attention_mask = _pin(self.attention_mask)
        self.position_embeddings = _pin(self.position_embeddings)
        
        return self
