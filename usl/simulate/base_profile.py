import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Tuple, Any, Union

from torch.utils.checkpoint import checkpoint
from transformers import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers import GPT2Config, LlamaConfig, Qwen2Config, Gemma2Config, AutoConfig


# TransformerBlock = Union[GPT2Block, LlamaDecoderLayer, Qwen3DecoderLayer, Gemma2DecoderLayer]


def _load_config(dir='data/models', model_name='gpt2-large') -> AutoConfig:
    if model_name == 'gpt2-large':
        fp = os.path.join(dir, 'gpt/gpt2-large/config.json')
    elif model_name == 'llama3.2-1b':
        fp = os.path.join(dir, 'meta-llama/llama3.2-1b/config.json')
    elif model_name == 'qwen3-0.6b':
        fp = os.path.join(dir, 'qwen/qwen3-0.6b/config.json')
    elif model_name == 'qwen3-1.7b':
        fp = os.path.join(dir, 'qwen/qwen3-1.7b/config.json')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return AutoConfig.from_pretrained(fp)


def _check_mem_alloc(device: torch.device = None):
    curr_alloc, max_alloc = torch.cuda.memory_allocated(device) / 1024**2, torch.cuda.max_memory_allocated(device) / 1024**2
    torch.cuda.reset_peak_memory_stats(device)
    return curr_alloc, max_alloc


def calculate_block_activation_size(block: TransformerBlock, with_checkpoint: bool = False, *args, **kwargs) -> int:
    """
    Calculate the total activation memory (in bytes) used by a forward pass.
    """
    hooks = []
    total_activation_size = [0]  # Use list for mutability in closure

    def extract_tensors(output: Any):
        if isinstance(output, torch.Tensor):
            return [output]
        elif isinstance(output, (list, tuple)):
            return [t for o in output for t in extract_tensors(o)]
        elif isinstance(output, dict):
            return [t for v in output.values() for t in extract_tensors(v)]
        return []

    def activation_hook(module, input, output: Tuple[torch.Tensor]):
        for o in extract_tensors(output):
            total_activation_size[0] += o.numel() * o.element_size()

    for module in block.modules():
        hooks.append(module.register_forward_hook(activation_hook))

    # block.train()
    if with_checkpoint:
        with torch.no_grad():
            output = block(*args, **kwargs)
    else:
        output = block(*args, **kwargs)

    for hook in hooks:
        hook.remove()

    return total_activation_size[0], output


def calculate_block_parameter_size(block: TransformerBlock) -> int:
    """
    Calculate the total parameter memory (in bytes) used by a model.
    """
    p_num = sum(p.numel() * p.element_size() for p in block.parameters())
    b_num = sum(p.numel() * p.element_size() for p in block.buffers())
    return p_num + b_num


def calculate_tensor_size(tensor: torch.Tensor = None) -> int:
    """
    Calculate the memory (in bytes) used by a tensor.
    """
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


class ProfLLMModel(nn.Module):
    def __init__(
        self,
        name='gpt2-large',
        layer: int = 1,
        hidden_size=1280,
    ):
        super().__init__()
        config = _load_config(model_name=name)
        if name == 'gpt2-large':
            config.n_layer = layer
            config.n_embd = hidden_size
            self.model = GPT2LMHeadModel(config=config)
        elif 'llama' in name:
            config.num_hidden_layers = layer
            config.hidden_size = hidden_size
            self.model = LlamaForCausalLM(config=config)
        elif 'qwen' in name:
            config.num_hidden_layers = layer
            config.hidden_size = hidden_size
            self.model = Qwen3ForCausalLM(config=config)
        # self.layers = nn.ModuleList(
        #     [GPT2Block(config=GPT2Config(hidden_size=hidden_size, num_attention_heads=num_attention_heads), layer_idx=i) for i in range(layer)]
        # )

    def forward(self, input, atten_mask, label):
        output = self.model(input_ids=input, attention_mask=atten_mask, labels=label)
        return output


def mem_profile(model='gpt2-large', layer=1, hidden_size=2560, mb_num=8):
    curr_alloc, max_alloc = _check_mem_alloc(device)
    print(f"[Before Model init]Current memory allocation: {curr_alloc:.2f} MB, Max memory allocation: {max_alloc:.2f} MB")
    model = ProfLLMModel(name=model, layer=layer, hidden_size=hidden_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    curr_alloc, max_alloc = _check_mem_alloc(device)
    print(f"[After Model init]Current memory allocation: {curr_alloc:.2f} MB, Max memory allocation: {max_alloc:.2f} MB")
    # model.train()
    # losses = []
    # for bid in range(2):
    #     print(f"----------------- Batch {bid} ------------------------")
    #     if bid == 1:
    #         curr_alloc, max_alloc = _check_mem_alloc(device)
    #         print(f"[Before Model forward]Current memory allocation: {curr_alloc:.2f} MB, Max memory allocation: {max_alloc:.2f} MB")
    #     for mb in range(mb_num):
    #         input_ids = torch.randint(0, 10000, (1, 512)).cuda()
    #         atten_mask = torch.ones(1, 512).cuda()
    #         output = model(input_ids, atten_mask=atten_mask, label=input_ids)
    #         losses.append(output.loss)
    #     if bid == 1:
    #         curr_alloc, max_alloc = _check_mem_alloc(device)
    #         print(f"[After Model forward]Current memory allocation: {curr_alloc:.2f} MB, Max memory allocation: {max_alloc:.2f} MB")
    #     for loss in losses:
    #         loss.backward()
    #     losses = []
    #     if bid == 1:
    #         curr_alloc, max_alloc = _check_mem_alloc(device)
    #         print(f"[After Model backward]Current memory allocation: {curr_alloc:.2f} MB, Max memory allocation: {max_alloc:.2f} MB")
    #     optimizer.step()
    #     optimizer.zero_grad(set_to_none=True)

    # return output


import argparse

if __name__ == "__main__":
    device = "cuda:2"
    torch.cuda.set_device(device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-M', type=str, default='gpt2-large')
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=2560)
    args = parser.parse_args()
    mem_profile(args.model, layer=args.layer, hidden_size=args.hidden_size)
