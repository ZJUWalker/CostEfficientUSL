from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2PreTrainedModel
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2RotaryEmbedding,
)
from usl.llm.split_config import SplitModelConfig


class Gemma2Server(Gemma2PreTrainedModel):

    def __init__(self, config: Gemma2Config, split_config):
        super().__init__(config)
        self.split_config = split_config
        self.layers: List[Gemma2DecoderLayer] = None
        self.rotary_emb = None

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Gemma2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Gemma2DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            # Gemma2 必须保留原始的 layer_idx 以支持 Sliding Window Attention
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])
        hidden_layers[i].is_sliding = False

        # 加载 RotaryEmbedding
        self.rotary_emb = pretrained_model.model.rotary_emb

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Gemma2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Gemma2DecoderLayer]

        # 物理初始化：注意这里传入的是绝对索引 i (from_l 到 to_l)，
        # 这一点对 Gemma2 至关重要，否则奇偶层的 Attention Window 模式会错乱
        self.layers = nn.ModuleList([Gemma2DecoderLayer(self.config, layer_idx=i) for i in range(from_l, to_l)])

        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
            self.layers[i - from_l].is_sliding = False

        # 加载 RotaryEmbedding
        self.rotary_emb = Gemma2RotaryEmbedding(config=pretrained_model.config)
        self.rotary_emb.load_state_dict(pretrained_model.model.rotary_emb.state_dict())

    def load_from_pretrained_model(self, pretrained_model: Gemma2ForCausalLM, logical=True):
        from_l = self.split_config.head_layer_num
        to_l = self.split_config.head_layer_num + self.split_config.server_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,  # 来自head的输出隐藏状态
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # 如果上游（Head）没有传 position_embeddings，这里需要重新计算
        # 因为 Gemma2 的 RoPE 是共享的，且需要在每层 Attention 中使用
        if position_embeddings is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            # Gemma2DecoderLayer forward 参数：
            # hidden_states, attention_mask, position_ids, past_key_values, ...
            # 注意：在 Transformers 实现中，如果没有传入 position_ids，
            # 只要传入了 position_embeddings 通常也能工作，或者层内部处理。
            # 这里我们保持最简调用。
            decoder_layer.is_sliding = False
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=None,  # 若已传入 position_embeddings，通常不需要显式 position_ids，或者由 rotary_emb 处理
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


def load_gemma2_server(pretrained_model: Gemma2ForCausalLM, split_config: SplitModelConfig) -> Gemma2Server:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.num_hidden_layers - split_config.head_layer_num - split_config.tail_layer_num

    gemma2_server = Gemma2Server(config, split_config)
    gemma2_server.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)

    return gemma2_server
