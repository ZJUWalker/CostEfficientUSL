import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
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
    dataset_name: str = "dialogsum",
    tokenizer: AutoTokenizer = None,
    client_ids: List[int] = [0],
    batch_size: int = 4,
    max_seq_len: int = 256,
    shuffle=False,
):
    # usl_dataset = get_dataset(dataset_name=dataset_name, tokenizer=tokenizer, client_ids=client_ids)
    client_dataloaders = get_client_dataloaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        client_ids=client_ids,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        splits=["train", "test"],
        shuffle=shuffle,
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


def manual_model_split(model: nn.Module, stage_index: int, num_stages: int, device: torch.device) -> Tuple[nn.Module, int, int, torch.device]:
    """
    手动将 Qwen3Server 拆成多个 stage：
    - 每个 stage 只保留自己负责的那一段 decoder layers
    - 其他层从 model.layers 中删掉
    - 返回一个 PipelineStage 封装后的模型

    注意：这个函数是原地修改 model 的，所以每个 rank 应该有自己的一份 model。
    """
    # 先确保模型在对应 device 上（避免后面参数在 CPU / CUDA 混合）
    model.to(device)

    total_layers = len(model.layers)
    if total_layers % num_stages != 0:
        # 如果不能整除，可以做个简单的 load balance：前几段多一层
        base = total_layers // num_stages
        extra = total_layers % num_stages

        # 计算当前 stage 的 [start, end)
        if stage_index < extra:
            # 前 extra 个 stage 每个多一层
            start = stage_index * (base + 1)
            end = start + (base + 1)
        else:
            start = extra * (base + 1) + (stage_index - extra) * base
            end = start + base
    else:
        # 能整除就均分
        per_stage = total_layers // num_stages
        start = stage_index * per_stage
        end = (stage_index + 1) * per_stage

    # ---- 删除不属于本 stage 的 layers ----
    # 注意：删除时一定要倒序删，否则下标会变化
    # 1) 删掉 [end, total_layers) 之后的层
    for i in reversed(range(end, total_layers)):
        del model.layers[i]

    # 2) 再删掉 [0, start) 之前的层
    for i in reversed(range(0, start)):
        del model.layers[i]

    # 此时 model.layers 的长度应该是 end - start
    assert len(model.layers) == end - start, f"Stage {stage_index}: layers count mismatch, expect {end - start}, got {len(model.layers)}"

    # Qwen3Server 里没有 tok_embeddings / norm / output 这种 head/tail，
    # 所以不需要像你示例里那样额外 del 这些模块。
    # rotary_emb 是所有层共用的，只保留一个没问题。

    # # 包装成 PipelineStage（按你示例里的方式）
    # stage = PipelineStage(
    #     model,
    #     stage_index,
    #     num_stages,
    #     device,
    # )
    return model


def load_stage_server_model(
    model_dir: str,
    model_name: str,
    split_point: int = 2,
    rank: int = 0,
    world_size: int = 1,
    use_lora: bool = False,
    use_qlora_4bit: bool = False,
    use_qlora_8bit: bool = False,
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
        server = get_peft_model(server, lora_config)

    # ---------------- 多卡切分 ----------------
    if world_size > 1:
        server = _split_server_into_stages(server, rank, world_size)

    return server


def _split_server_into_stages(server: nn.Module, rank: int, world_size: int) -> nn.Module:
    # 假设 server.layers 是 ModuleList
    layers = list(server.layers)
    total_layers = len(layers)

    layers_per_stage = total_layers // world_size
    remainder = total_layers % world_size

    if rank < remainder:
        start_layer = rank * (layers_per_stage + 1)
        end_layer = start_layer + layers_per_stage + 1
    else:
        start_layer = remainder * (layers_per_stage + 1) + (rank - remainder) * layers_per_stage
        end_layer = start_layer + layers_per_stage

    stage_layers = layers[start_layer:end_layer]

    # 不要再用裸 nn.Module，而是用我们定义好的 StageModel
    # stage_model = StageModel(stage_layers, start_layer=start_layer, end_layer=end_layer)
    stage_model = nn.ModuleList(stage_layers)

    # ❗通常不建议在这里 `del layers`，因为会直接改掉原始 server.layers
    # 如果你确实需要修改原模型，可以按需保留
    # del server.layers[:start_layer]
    # del server.layers[end_layer:]

    return stage_model
