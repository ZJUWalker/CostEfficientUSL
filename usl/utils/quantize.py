import torch
from typing import Tuple, Union, Optional


def quantize(x: torch.Tensor, n_bits=8) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if n_bits == 8:
        return quantize_to_int8(x)
    elif n_bits == 4:
        return quantize_to_int4(x)
    elif n_bits == 2:
        return quantize_to_2bit(x)
    else:
        return x, None  # X,scale=None


def dequantize(x: torch.Tensor, scale: torch.Tensor, n_bits=8) -> torch.Tensor:
    if n_bits == 8:
        return dequantize_from_int8(x, scale)
    elif n_bits == 4:
        return dequantize_from_int4(x, scale)
    elif n_bits == 2:
        return dequantize_from_2bit(x, scale)
    else:
        return x


@torch.no_grad()
def quantize_to_int8(tensor: torch.Tensor):
    """
    将 fp32 tensor 压缩为 int8
    返回: (int8_tensor, scale_float)
    """
    # 1. 计算最大绝对值 (AbsMax)
    # 加上 1e-6 防止除以 0
    max_val = tensor.abs().max() + 1e-6

    # 2. 计算缩放因子 scale
    # 映射目标是 [-127, 127] (保留 -128 作为一个特殊值或直接忽略)
    scale = 127.0 / max_val

    # 3. 量化
    # round() 四舍五入，clamp() 截断防止溢出，最后转为 int8
    tensor_int8 = (tensor * scale).round().clamp(-127, 127).to(torch.int8)
    scale = scale.to('cpu')
    return tensor_int8, scale


@torch.no_grad()
def dequantize_from_int8(tensor_int8: torch.Tensor, scale: torch.Tensor, original_dtype=torch.float32):
    """
    将 int8 tensor 解压为 fp32
    """
    # 1. 还原
    scale = scale.to(tensor_int8.device, dtype=original_dtype)
    tensor_fp32 = tensor_int8.to(original_dtype) / scale

    return tensor_fp32


@torch.no_grad()
def quantize_to_int4(x: torch.Tensor, scale_method='max', scale_dims=(0, 1)):

    x, scale = _compress_nbits(x, bits=4, scale_method=scale_method, scale_dims=scale_dims)

    x0, x1 = x.chunk(2, -1)
    x = (x0 << 4) + x1

    return x, scale


@torch.no_grad()
def dequantize_from_int4(x: torch.Tensor, scale: float):

    bitmask = 15

    x0 = x >> 4
    x1 = x & bitmask

    x = torch.cat([x0, x1], -1)

    x = _decompress_nbits(x, scale, bits=4)

    return x


@torch.no_grad()
def quantize_to_2bit(x: torch.Tensor, scale_method='max', scale_dims=(0, 1)):

    x, scale = _compress_nbits(x, bits=2, scale_method=scale_method, scale_dims=scale_dims)

    x0, x1, x2, x3 = x.chunk(4, -1)
    x = (x0 << 6) + (x1 << 4) + (x2 << 2) + x3

    return x, scale


@torch.no_grad()
def dequantize_from_2bit(x: torch.Tensor, scale: float):

    bitmask = 3

    x0 = x >> 6
    x1 = (x >> 4) & bitmask
    x2 = (x >> 2) & bitmask
    x3 = x & bitmask
    x = torch.cat([x0, x1, x2, x3], -1)

    x = _decompress_nbits(x, scale, bits=2)

    return x


@torch.no_grad()
def _rounding(x: torch.Tensor, stochastic=False, minimum_stochastic_distance=0.2):
    if stochastic:
        x_floor = x.floor()
        th = x - x_floor
        if minimum_stochastic_distance > 0:
            th[th < minimum_stochastic_distance] = 0.0
            th[th > 1 - minimum_stochastic_distance] = 1.0
        pr = torch.rand_like(x)
        x_floor += pr < th
        return x_floor
    else:
        return x.round()


@torch.no_grad()
def _compress_nbits(x: torch.Tensor, bits: int, scale_method='max', scale_dims=(0, 1)):

    fbits = bits - 1

    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax(scale_dims, keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean(scale_dims, keepdims=True).sqrt() * 2
    else:
        raise Exception('unkonwn scale method.')
    # fp16 should be enough
    scale: torch.Tensor = scale.half()
    x = x / (scale + 1e-6)

    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1 << fbits)
    clip_max = (1 << fbits) - 1

    x = _rounding(x)
    x = x.clip(clip_min, clip_max)

    x = x - clip_min
    x = x.type(torch.uint8)
    scale = scale.to('cpu')

    return x, scale


@torch.no_grad()
def _decompress_nbits(x: torch.Tensor, scale: torch.Tensor, bits: int):
    scale = scale.to(x.device, dtype=x.dtype)
    fbits = bits - 1

    clip_min = -(1 << fbits)
    clip_max = (1 << fbits) - 1

    x = x.float() + clip_min

    x = x / (clip_max + 1) * scale

    return x


# --- 测试主程序 ---
if __name__ == "__main__":
    import torch.nn.functional as F

    print("🚀 开始测试 2-bit 压缩与解压...")

    # 1. 创建模拟数据 (模拟激活值)
    # 假设 batch_size=2, seq_len=8, hidden_dim=16 (hidden_dim 必须能被 4 整除，因为我们要打包 4 个数)
    original_data = torch.randn(2, 8, 16, requires_grad=False) * 2
    print(f"原始数据形状: {original_data.shape}")
    print(f"原始数据范围: [{original_data.min():.3f}, {original_data.max():.3f}]")

    # 2. 执行压缩
    compressed_tensor, scale_param = quantize_to_int4(original_data, scale_dims=(0, 1))

    print(f"\n📦 压缩后数据形状: {compressed_tensor.shape}")
    print(
        f"压缩比: {original_data.numel()*original_data.element_size() / compressed_tensor.numel()*compressed_tensor.element_size():.2f} : 1 (理论上显存占用减少了 16 倍)"
    )
    print(f"Scale 参数形状: {scale_param.shape}")

    # 3. 执行解压
    restored_data = dequantize_from_int4(compressed_tensor, scale_param)

    print(f"\n🔧 解压后数据形状: {restored_data.shape}")

    # 4. 计算误差
    mse = F.mse_loss(original_data, restored_data).item()
    psnr = 10 * torch.log10((original_data.max() ** 2) / mse) if mse > 0 else float('inf')

    print(f"均方误差: {mse:.6f}")
    print(f"峰值信噪比: {psnr:.2f} dB")

    # 5. 视觉对比 (打印一小部分)
    print(f"\n🔍 数据对比 (前 10 个值):")
    print(f"原始: {original_data.flatten()[:10].cpu().numpy()}")
    print(f"解压: {restored_data.flatten()[:10].cpu().numpy()}")

    print("\n 测试完成！")
