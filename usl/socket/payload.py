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
