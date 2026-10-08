"""Hugging Face model loading followed by generic logical partitioning."""

from __future__ import annotations

from typing import Literal

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from usl_ray.config import ExperimentConfig
from usl_ray.models.partition import DecoderPartition, SplitPlan, partition_model

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_full_model(config: ExperimentConfig) -> nn.Module:
    """Load or randomly initialize the configured causal language model on CPU."""
    dtype = _DTYPES[config.model.dtype]
    hf_config = AutoConfig.from_pretrained(config.model.path, local_files_only=True)
    hf_config._attn_implementation = config.model.attn_implementation
    if config.model.init == "random":
        return AutoModelForCausalLM.from_config(
            hf_config, dtype=dtype, attn_implementation=config.model.attn_implementation
        )
    return AutoModelForCausalLM.from_pretrained(
        config.model.path,
        dtype=dtype,
        attn_implementation=config.model.attn_implementation,
        device_map="cpu",
        local_files_only=True,
    )


def load_tokenizer(config: ExperimentConfig) -> PreTrainedTokenizerBase:
    """Load a right-padding tokenizer from local files."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.path, local_files_only=True)
    tokenizer.padding_side = config.batch.padding_side
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _add_lora(partition: DecoderPartition, config: ExperimentConfig) -> DecoderPartition:
    lora = config.model.lora
    if not lora.enabled:
        return partition
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
    )
    return get_peft_model(partition, peft_config)  # type: ignore[return-value]


def load_client_partitions(config: ExperimentConfig) -> tuple[DecoderPartition, DecoderPartition]:
    """Load head and tail from one model so tied weights remain shared."""
    model = load_full_model(config)
    plan = SplitPlan(config.model.num_hidden_layers, config.split.split_point)
    head = partition_model(model, config.model.architecture, plan.head, include_embedding=True)
    tail = partition_model(model, config.model.architecture, plan.tail, include_output=True)
    return _add_lora(head, config), _add_lora(tail, config)


def load_server_partition(config: ExperimentConfig, rank: int) -> DecoderPartition:
    """Load the decoder layers assigned to one server rank."""
    if config.server is None:
        raise ValueError("server config is required")
    model = load_full_model(config)
    plan = SplitPlan(config.model.num_hidden_layers, config.split.split_point)
    ranges = plan.server_stages(config.server.world_size)
    if rank < 0 or rank >= len(ranges):
        raise ValueError(f"Server rank {rank} is outside [0, {len(ranges)})")
    partition = partition_model(model, config.model.architecture, ranges[rank])
    return _add_lora(partition, config)


def partition_role(
    config: ExperimentConfig,
    role: Literal["head", "tail", "server"],
    rank: int = 0,
) -> DecoderPartition:
    """Convenience loader used by launchers and tests."""
    if role == "server":
        return load_server_partition(config, rank)
    head, tail = load_client_partitions(config)
    return head if role == "head" else tail
