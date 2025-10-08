from dataclasses import dataclass
from typing import Any, Tuple, Literal
import torch


@dataclass
class Payload:
    tensor: torch.Tensor  # `Any` 适用于 tensor 类型，可以根据具体的类型进行调整
    is_activation: bool = True
    phase: str = Literal["FWD", "BWD"]
    token: str = ""
    group_id: str = ""
    mb_idx: int = 0
    mb_total: int = 0
    attention_mask: torch.Tensor = None
    position_embeddings: Tuple[torch.Tensor, ...] = None

    def payload_nbytes(self) -> int:
        """计算 payload 中所有 tensor 的占用字节数（单位: Byte）"""
        total = 0

        def tensor_nbytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        try:
            for val in [self.tensor, self.attention_mask, self.position_embeddings]:
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    total += tensor_nbytes(val)
                elif isinstance(val, (list, tuple)):
                    for v in val:
                        if isinstance(v, torch.Tensor):
                            total += tensor_nbytes(v)
        except Exception as e:
            return 0
        return total
