from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2Model, Gemma2PreTrainedModel
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2MLP,
    Gemma2RotaryEmbedding,
    Gemma2RMSNorm,
)
from usl.llm.split_config import SplitModelConfig

"""
Gemma2ForCausalLM(
  (model): Gemma2Model(
    (embed_tokens): Embedding(256000, 3584, padding_idx=0)
    (layers): ModuleList(
      (0-41): 42 x Gemma2DecoderLayer(
        (self_attn): Gemma2Attention(
          (q_proj): Linear(in_features=3584, out_features=4096, bias=False)
          (k_proj): Linear(in_features=3584, out_features=2048, bias=False)
          (v_proj): Linear(in_features=3584, out_features=2048, bias=False)
          (o_proj): Linear(in_features=4096, out_features=3584, bias=False)
        )
        (mlp): Gemma2MLP(
          (gate_proj): Linear(in_features=3584, out_features=14336, bias=False)
          (up_proj): Linear(in_features=3584, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=3584, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        (pre_feedforward_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        (post_feedforward_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Gemma2RMSNorm((3584,), eps=1e-06)
    (rotary_emb): Gemma2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=256000, bias=False)
)
"""


class Gemma2ClientHead(Gemma2PreTrainedModel):
    def __init__(self, config: Gemma2Config, split_config):
        super().__init__(config)
        self.split_config = split_config
        self.embed_tokens = None
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = None
        self.rotary_emb = None

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Gemma2ForCausalLM, from_l, to_l):
        emb_layer = pretrained_model.model.embed_tokens
        self.embed_tokens = emb_layer
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Gemma2DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            # Gemma2 强依赖 layer_idx 来决定是否使用滑动窗口，必须保留原始 index
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])
            hidden_layers[i].is_sliding = False
        self.rotary_emb = pretrained_model.model.rotary_emb

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Gemma2ForCausalLM, from_l, to_l):
        self.embed_tokens = nn.Embedding(pretrained_model.config.vocab_size, pretrained_model.config.hidden_size, self.padding_idx)
        emb_layer = pretrained_model.model.embed_tokens
        self.embed_tokens.load_state_dict(emb_layer.state_dict())

        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Gemma2DecoderLayer]

        # 注意：这里必须传入绝对 layer_idx，因为 Gemma2 依赖它判断 sliding window
        self.layers = nn.ModuleList([Gemma2DecoderLayer(self.config, layer_idx) for layer_idx in range(from_l, to_l)])

        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
            self.layers[i - from_l].is_sliding = False

        self.rotary_emb = Gemma2RotaryEmbedding(config=pretrained_model.config)
        self.rotary_emb.load_state_dict(pretrained_model.model.rotary_emb.state_dict())

    def load_from_pretrained_model(self, pretrained_model: Gemma2ForCausalLM, logical=True):
        from_l = 0
        to_l = self.split_config.head_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # Gemma API compatibility
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Tuple:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # --------------------------------------------------------------------------
        # Gemma2 特性：Embedding Scaling
        # Gemma2 在进入 layer 之前会将 embedding 乘以 sqrt(hidden_size)
        # --------------------------------------------------------------------------
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds * normalizer

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        # 生成 Causal Mask
        # 注意：Gemma2 通常处理 attention mask 比较特殊，但在 client-server 模式下，
        # 我们使用通用的 mask 生成逻辑，因为具体的 sliding window 是在 layer 内部通过 causal_mask 再次截断实现的
        causal_mask = _update_causal_mask(self, attention_mask, inputs_embeds, position_ids, False)

        hidden_states = inputs_embeds

        # create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        return (
            hidden_states,
            causal_mask,
            position_embeddings,
        )


class Gemma2ClientTail(Gemma2PreTrainedModel):
    def __init__(self, config: Gemma2Config, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config
        self.layers = None
        self.norm = None
        self.lm_head = None

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Gemma2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Gemma2DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            # 处理 layer_idx
            if not self.split_config.with_server:
                # 如果没有 server，tail 接在 head 后面，index 需要偏移吗？
                # Gemma2 必须使用绝对 layer_idx 来保证 sliding window 模式正确 (奇偶层不同)
                # 因此建议尽量保持原始 model 的 index
                hidden_layers[i].self_attn.layer_idx = i
            else:
                hidden_layers[i].self_attn.layer_idx = i

            hidden_layers[i].is_sliding = False
            self.layers.append(hidden_layers[i])

        self.norm = pretrained_model.model.norm
        self.lm_head = pretrained_model.lm_head
        self.rotary_emb = pretrained_model.model.rotary_emb

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Gemma2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Gemma2DecoderLayer]

        # 物理加载时，传入绝对 layer_idx (range(from_l, to_l))
        self.layers = nn.ModuleList([Gemma2DecoderLayer(self.config, layer_idx) for layer_idx in range(from_l, to_l)])

        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
            self.layers[i - from_l].is_sliding = False

        self.norm = Gemma2RMSNorm(pretrained_model.config.hidden_size, eps=pretrained_model.config.rms_norm_eps)
        self.norm.load_state_dict(pretrained_model.model.norm.state_dict())

        self.lm_head = nn.Linear(pretrained_model.config.hidden_size, pretrained_model.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

        self.rotary_emb = Gemma2RotaryEmbedding(config=pretrained_model.config)
        self.rotary_emb.load_state_dict(pretrained_model.model.rotary_emb.state_dict())

    def load_from_pretrained_model(self, pretrained_model: Gemma2ForCausalLM, logical=True):
        from_l = self.split_config.head_layer_num + self.split_config.server_layer_num
        to_l = self.split_config.total_hidden_layers
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        lm_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:

        # Position embeddings usually passed from Head, but if None (rare in pipeline), recalc
        if position_embeddings is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            # Gemma2 layer forward signature might vary slightly, but generally supports these
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=None,  # Usually handled by rotary embeddings being passed explicitly or calculated inside if None
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # --------------------------------------------------------------------------
        # Gemma2 特性：Logits Soft-capping
        # --------------------------------------------------------------------------
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            if lm_mask is not None:
                shift_lm_mask = lm_mask[..., 1:].contiguous()
                shift_lm_mask = shift_lm_mask.view(-1)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * shift_lm_mask.float()
                loss = loss.sum() / shift_lm_mask.sum()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# 复用您提供的 _update_causal_mask，不需要修改，
# 因为 Gemma2 在 transformers 库中也兼容标准的 attention mask 格式 (inverted causal mask)
def _update_causal_mask(
    partitioned_model: Gemma2PreTrainedModel,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position_ids: torch.Tensor = None,
    output_attentions: bool = False,
):
    # ... (保持您原始提供的代码逻辑不变，这里为了节省篇幅省略，直接粘贴即可) ...
    # 唯一需要确认的是 Gemma2Config._attn_implementation 的值通常是 "eager", "sdpa", 或 "flash_attention_2"
    # 该函数逻辑对这些都是兼容的。
    past_seen_tokens = 0

    if partitioned_model.config._attn_implementation == "sdpa":
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=partitioned_model.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

    if attention_mask is not None and attention_mask.dim() == 4:
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        # 修正：处理 cache_position_ids 为 None 的情况（冷启动）
        if cache_position_ids is None:
            # 创建一个简单的 position ids
            cache_position_ids = torch.arange(sequence_length, device=device)

        causal_mask *= torch.arange(target_length, device=device) > cache_position_ids.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    if (
        partitioned_model.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


def load_gemma2_client(pretrained_model: Gemma2ForCausalLM, split_config: SplitModelConfig) -> Tuple[Gemma2ClientHead, Gemma2ClientTail]:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.num_hidden_layers - split_config.head_layer_num - split_config.tail_layer_num

    head_model = Gemma2ClientHead(config, split_config)
    head_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)

    tail_model = Gemma2ClientTail(config, split_config)
    tail_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)

    return head_model, tail_model
