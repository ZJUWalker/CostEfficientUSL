import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

# ==========================================
# 1. Generating NormalFloat(NF)
# ==========================================
def generate_nf_table(bits: int, device="cpu", dtype=torch.float16):
    """Generate a b-bit discrete codebook for NormalFloat."""
    n_levels = 2 ** bits
    p = (np.arange(n_levels) + 0.5) / n_levels
    q = norm.ppf(p)
    q = q / np.max(np.abs(q))
    q = torch.tensor(q).to(device=device, dtype=dtype)
    return q

# ==========================================
# 2. Bit-packing
# ==========================================
def pack_2bit_tensor(q: torch.Tensor):
    """Compress to uint8, reducing the size by 4."""
    assert q.dtype == torch.uint8
    q_shape = q.shape
    q = q.reshape(-1)

    pad = (-q.numel()) % 4
    if pad:
        q = torch.cat([q, torch.zeros(pad, dtype=q.dtype, device=q.device)])
    else:
        pad = 0

    q = q.view(-1, 4)
    packed = (
        (q[:, 0] << 0) |
        (q[:, 1] << 2) |
        (q[:, 2] << 4) |
        (q[:, 3] << 6)
    ).to(torch.uint8)

    return packed, q_shape, pad

def unpack_2bit_tensor(packed: torch.Tensor, q_shape, pad=0):
    assert packed.dtype == torch.uint8
    q = torch.empty((packed.numel(), 4), dtype=torch.uint8, device=packed.device)
    q[:, 0] = (packed >> 0) & 0b11
    q[:, 1] = (packed >> 2) & 0b11
    q[:, 2] = (packed >> 4) & 0b11
    q[:, 3] = (packed >> 6) & 0b11

    q = q.view(-1)
    if pad:
        q = q[:-pad]
    q = q.reshape(q_shape)
    return q.float()

def pack_4bit_tensor(q: torch.Tensor):
    assert q.dtype == torch.uint8
    q_shape = q.shape
    q = q.reshape(-1)

    pad = (-q.numel()) % 2
    if pad:
        q = torch.cat([q, torch.zeros(pad, dtype=q.dtype, device=q.device)])
    else:
        pad = 0

    q = q.view(-1, 2)
    packed = (
        (q[:, 0] & 0x0F) |
        ((q[:, 1] & 0x0F) << 4)
    ).to(torch.uint8)

    return packed, q_shape, pad

def unpack_4bit_tensor(packed: torch.Tensor, q_shape, pad=0):
    assert packed.dtype == torch.uint8
    q = torch.empty((packed.numel(), 2), dtype=torch.uint8, device=packed.device)
    q[:, 0] = packed & 0x0F
    q[:, 1] = (packed >> 4) & 0x0F

    q = q.view(-1)
    if pad:
        q = q[:-pad]
    return q.reshape(q_shape).float()


# ==========================================
# 3. Double Quantization
# ==========================================
class QLoRACommQuantizer(nn.Module):
    """
    based on QLoRA:
    FWD: NormalFloat (NF4/NF2)
    BWD: per-block + INT8/INT4 quantization
    """
    def __init__(self, activation_bits=4, gradient_bits=8, block_size=64, use_double_quant=True):
        super().__init__()
        self.act_bits = activation_bits
        self.grad_bits = gradient_bits
        self.block_size = block_size
        self.use_double_quant = use_double_quant
        
        supported_bits = [2, 4, 8]
        if self.act_bits not in supported_bits or self.grad_bits not in supported_bits:
            raise ValueError(f"Unsupported bit width. Supported widths are: {supported_bits}")
            
        if self.act_bits < 8:
            self.table = generate_nf_table(self.act_bits)
        else:
            self.table = None

    def compress(self, x: torch.Tensor, mode: str = "activation"):
        assert mode in ["activation", "gradient"], f"Invalid mode: {mode}"
        
        # 动态获取当前阶段对应的位宽
        current_bits = self.act_bits if mode == "activation" else self.grad_bits
        
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        x_shape = x_flat.shape
        
        # block-wise
        x_blocked = x_flat.view(x_shape[0], x_shape[1] // self.block_size, self.block_size)

        if mode == "activation":
            x_min = x_blocked.min(dim=2).values.unsqueeze(-1)
            x_max = x_blocked.max(dim=2).values.unsqueeze(-1)
            scales = (x_max - x_min).squeeze(-1)

            if self.table is not None:
                x_norm = 2 * (x_blocked - x_min) / (x_max - x_min + 1e-8) - 1
                dist = torch.abs(x_norm.unsqueeze(-1) - self.table.to(x.device))
                q_idx = torch.argmin(dist, dim=-1).to(torch.uint8)
            else:
                # Fallback: 如果前向配置了 8-bit，则使用线性映射
                step = (x_max - x_min) / 255.0
                q_idx = torch.round((x_blocked - x_min) / (step + 1e-8)).clamp(0, 255).to(torch.uint8)
                
            aux_specific = {"mins": x_min}
            
        else:
            # for BWD gradient
            abs_max = x_blocked.abs().max(dim=2).values.unsqueeze(-1)
            scales = abs_max.squeeze(-1)
            
            q_min = -(1 << (self.grad_bits - 1))
            q_max = (1 << (self.grad_bits - 1)) - 1
            
            step_size = (abs_max / q_max).clamp(min=1e-12)
            
            x_q = torch.round(x_blocked / step_size)
            x_q = torch.clamp(x_q, q_min, q_max)
            
            q_idx = (x_q - q_min).to(torch.uint8)
            
            aux_specific = {}

        # Double Quantization
        if self.use_double_quant:
            s_min = scales.min(dim=-1).values.unsqueeze(-1)
            s_max = scales.max(dim=-1).values.unsqueeze(-1)
            scales_q = ((scales - s_min) / (s_max - s_min + 1e-8) * 255).round().to(torch.uint8)
        else:
            scales_q, s_min, s_max = None, scales, None

        # Bit-packing
        if current_bits == 2:
            packed, q_shape, pad = pack_2bit_tensor(q_idx)
        elif current_bits == 4:
            packed, q_shape, pad = pack_4bit_tensor(q_idx)
        elif current_bits == 8:
            packed, q_shape, pad = q_idx, q_idx.shape, 0

        payload = {
            "packed_indices": packed,
            "pad": pad,
            "q_shape": q_shape
        }
        
        aux = {
            "scales_q": scales_q,
            "s_min": s_min,
            "s_max": s_max,
            "x_shape": x_shape,
            "original_shape": original_shape,
            **aux_specific
        }

        return payload, aux

    def decompress(self, payload: dict, aux: dict, mode: str = "activation"):
        packed = payload["packed_indices"]
        pad = payload["pad"]
        q_shape = payload["q_shape"]
        
        current_bits = self.act_bits if mode == "activation" else self.grad_bits
        
        if current_bits == 2:
            q_idx = unpack_2bit_tensor(packed, q_shape, pad).to(packed.device)
        elif current_bits == 4:
            q_idx = unpack_4bit_tensor(packed, q_shape, pad).to(packed.device)
        elif current_bits == 8:
            # ======= [新增 .view(q_shape) 作为安全屏障] =======
            # 这样如果 Server 错误地传来了 4-bit(1D Tensor)，这里会立刻抛出明确的维度报错
            # 而不会进入后续的矩阵乘法导致内存爆炸
            q_idx = packed.to(packed.device).view(q_shape)

        scales_q, s_min, s_max = aux["scales_q"], aux["s_min"], aux["s_max"]
        if scales_q is not None:
            scales = s_min + (scales_q.float() / 255) * (s_max - s_min)
        else:
            scales = s_min
        scales = scales.unsqueeze(-1)

        x_shape, original_shape = aux["x_shape"], aux["original_shape"]

        if mode == "activation":
            mins = aux["mins"]
            if self.table is not None:
                w_block = self.table[q_idx.long()].to(dtype=scales.dtype, device=scales.device)
                w_block = (w_block + 1) / 2 * scales + mins
            else:
                # 8-bit fallback
                w_block = (q_idx.float() / 255.0) * scales + mins
        else:
            q_min = -(1 << (self.grad_bits - 1))
            q_max = (1 << (self.grad_bits - 1)) - 1
            
            x_q = q_idx.float() + q_min
            
            step_size = (scales / q_max).clamp(min=1e-12)
            w_block = x_q * step_size
            
        flatten_x = w_block.view(x_shape)
        return flatten_x.view(original_shape)
