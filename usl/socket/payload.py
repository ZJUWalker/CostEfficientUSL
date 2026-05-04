from dataclasses import dataclass, field
from contextlib import nullcontext
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


@dataclass
class StagedPayload:
    payload: Payload
    ready_event: Optional[torch.cuda.Event] = None
    hold_refs: Tuple[Any, ...] = field(default_factory=tuple)


def _stage_obj_to_cpu(obj: Any, hold_refs: list[Any]) -> Any:
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach()
        if tensor.is_cuda:
            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor, non_blocking=True)
            hold_refs.append(tensor)
            return cpu_tensor
        return tensor
    if isinstance(obj, dict):
        return {k: _stage_obj_to_cpu(v, hold_refs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_stage_obj_to_cpu(v, hold_refs) for v in obj)
    return obj


def stage_payload_for_transfer(
    payload: Payload,
    *,
    copy_stream: Optional[torch.cuda.Stream] = None,
    wait_event: Optional[torch.cuda.Event] = None,
) -> StagedPayload:
    hold_refs: list[Any] = []
    active_stream = copy_stream
    copy_ctx = torch.cuda.stream(active_stream) if active_stream is not None else nullcontext()

    with copy_ctx:
        if wait_event is not None and active_stream is not None:
            active_stream.wait_event(wait_event)

        staged_payload = Payload(
            tensor=_stage_obj_to_cpu(payload.tensor, hold_refs),
            aux=_stage_obj_to_cpu(payload.aux, hold_refs) if payload.aux is not None else None,
            is_compressed=payload.is_compressed,
            is_activation=payload.is_activation,
            phase=payload.phase,
            token=payload.token,
            group_id=payload.group_id,
            mb_idx=payload.mb_idx,
            mb_total=payload.mb_total,
            attention_mask=_stage_obj_to_cpu(payload.attention_mask, hold_refs),
            position_embeddings=_stage_obj_to_cpu(payload.position_embeddings, hold_refs),
        )

        ready_event = None
        if hold_refs and active_stream is not None:
            ready_event = torch.cuda.Event()
            active_stream.record_event(ready_event)

    return StagedPayload(payload=staged_payload, ready_event=ready_event, hold_refs=tuple(hold_refs))
