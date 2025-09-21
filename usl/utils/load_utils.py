import torch
import torch.nn as nn
from typing import Dict, Any, List
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from usl.llm import (
    load_gpt_server_model,
    load_llama_server,
    load_qwen3_server,
    SplitModelConfig,
)
from usl.llm import (
    load_gpt_client_models,
    load_llama_client,
    load_qwen3_client,
    SplitModelConfig,
)
from usl.utils.dataset.base import get_client_dataloaders
from usl.utils.dataset.exp import get_dataset


def get_model_layer_num(model_dir: str) -> int:
    config = AutoConfig.from_pretrained(model_dir)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):
        return config.n_layer
    else:
        raise ValueError("Cannot find layer number")


def _load_orginal_model(
    model_dir: str,
    split_point: int = 2,
    use_qlora_4bit=False,
    use_qlora_8bit=False,
):
    if use_qlora_4bit or use_qlora_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_qlora_4bit,
            load_in_8bit=use_qlora_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=quantization_config, device_map="cpu")

    if use_qlora_4bit or use_qlora_8bit:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    split_config = SplitModelConfig(
        head_layer_num=split_point,
        tail_layer_num=split_point,
    )
    return model, tokenizer, split_config


def load_client(
    model_dir: str,
    model_name: str,
    split_point: int = 2,
    use_lora=False,
    use_qlora_4bit=False,
    use_qlora_8bit=False,
):
    # ---------------- 加载模型 ----------------
    model, tokenizer, split_config = _load_orginal_model(
        model_dir,
        split_point=split_point,
        use_qlora_4bit=use_qlora_4bit,
        use_qlora_8bit=use_qlora_8bit,
    )

    # ---------------- 按模型类型加载 ----------------
    if "gpt" in model_name.lower():
        head, tail = load_gpt_client_models(model, split_config)
    elif "llama" in model_name.lower():
        head, tail = load_llama_client(model, split_config)
    elif "qwen" in model_name.lower():
        head, tail = load_qwen3_client(model, split_config)
    else:
        raise ValueError(f"unsupported model card {model_name}")
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        head = get_peft_model(head, lora_config)
        tail = get_peft_model(tail, lora_config)
    del model
    return head, tail, tokenizer


def load_dataset(
    dataset_name: str = "gsm8k", tokenizer: AutoTokenizer = None, client_ids: List[int] = [0], batch_size: int = 4, max_seq_len: int = 256
):
    # usl_dataset = get_dataset(dataset_name=dataset_name, tokenizer=tokenizer, client_ids=client_ids)
    client_dataloaders = get_client_dataloaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        client_ids=client_ids,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        splits=["train", "test"],
        shuffle=False,
    )
    return client_dataloaders


def load_server_model(
    model_dir: str,
    model_name: str,
    split_point: int = 2,
    use_lora=False,
    use_qlora_4bit=False,
    use_qlora_8bit=False,
) -> nn.Module:

    # ---------------- 加载模型 ----------------
    model, _, split_config = _load_orginal_model(
        model_dir,
        split_point=split_point,
        use_qlora_4bit=use_qlora_4bit,
        use_qlora_8bit=use_qlora_8bit,
    )

    # ---------------- 按模型类型加载 ----------------
    if "gpt" in model_name.lower():
        server = load_gpt_server_model(model, split_config)
    elif "llama" in model_name.lower():
        server = load_llama_server(model, split_config)
    elif "qwen" in model_name.lower():
        server = load_qwen3_server(model, split_config)
    else:
        raise ValueError(f"unsupported model card {model_name}")

    # ---------------- LoRA 配置 ----------------
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    return server
