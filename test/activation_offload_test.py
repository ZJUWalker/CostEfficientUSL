import time
import torch
import torch.nn as nn
from usl.offload.activation_offload import (
    AsyncDoubleBufferGroupOffloadHandler,
    CpuOffloadHookWithOffloadHandler,
    SynchronizedGroupOffloadHandler,
)
from usl.offload import ModelParamOffload, OptimizerStateOffload
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import contextlib

from usl.utils.load_utils import load_dataset

mini_batch_num = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_model_memory(model: nn.Module):
    total_params = 0
    for param in model.parameters():
        # 参数的形状，计算每个参数的大小
        total_params += param.numel() * param.element_size()  # element_size() gives the size of each element in bytes
    # 将字节转换为MB
    memory_in_mb = total_params / (1024**2)
    return memory_in_mb


def _get_memory_usage_mb():
    return round(torch.cuda.memory_allocated() / 1024**2, 4), round(torch.cuda.max_memory_allocated() / 1024**2, 4)


def _init():
    torch.manual_seed(42)
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.Stream(device))
    print("init", _get_memory_usage_mb(), "MB")
    head_model = GPT2LMHeadModel(GPT2Config(n_embd=1280, n_head=20, n_inner=4096, n_layer=3)).to(device)
    # head_model.train()
    print("head model memory", _get_memory_usage_mb(), "MB")
    tail_model = GPT2LMHeadModel(GPT2Config(n_embd=1280, n_head=20, n_inner=4096, n_layer=3)).to(device)
    # tail_model.train()
    head_optimizer = torch.optim.Adam(head_model.parameters(), lr=0.001)
    tail_optimizer = torch.optim.Adam(tail_model.parameters(), lr=0.001)
    tokenizer = GPT2Tokenizer.from_pretrained("data/models/gpt/gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    client_dataloaders = load_dataset(tokenizer=tokenizer, client_ids=[0], batch_size=4, max_seq_len=256)
    dataloader = client_dataloaders[0]['train']  # 默认只取第一个客户端数据
    return head_model, tail_model, head_optimizer, tail_optimizer, tokenizer, dataloader


def test_no_offload():
    head_model, tail_model, head_optimizer, tail_optimizer, tokenizer, dataloader = _init()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=0),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
        on_trace_ready=(torch.profiler.tensorboard_trace_handler("./log/trace", worker_name="no_offload")),  # 保存到 TensorBoard
        # on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            head_losses = []
            tail_losses = []
            print("total iter------", i + 1)
            # head fwd
            print("head fwd")
            for idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids
                output = head_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                head_losses.append(loss)
                if idx == mini_batch_num - 1:
                    break
            # tail fwd
            # time.sleep(0.5)
            print("tail fwd")
            for idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids
                output = tail_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                tail_losses.append(loss)
                tail_losses[idx].backward()
                if idx == mini_batch_num - 1:
                    break
            tail_optimizer.step()
            tail_optimizer.zero_grad()  # 优化器梯度归零
            # head bwd,simulate time consuming
            # time.sleep(0.5)
            print("head bwd")
            for j in range(mini_batch_num):
                head_losses[j].backward()
            head_optimizer.step()
            head_optimizer.zero_grad()  # 优化器梯度归零
            prof.step()
            torch.cuda.reset_peak_memory_stats()


def test_activation_offload():
    head_model, tail_model, head_optimizer, tail_optimizer, tokenizer, dataloader = _init()
    head_cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(num_minibatch=mini_batch_num)
    head_cpu_offload_context = CpuOffloadHookWithOffloadHandler(head_cpu_offload_handler)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=0),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
        on_trace_ready=(torch.profiler.tensorboard_trace_handler("./log/trace", worker_name="activation_offload")),  # 保存到 TensorBoard
        # on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            head_losses = []
            tail_losses = []
            print("total iter------", i + 1)
            # head fwd
            print("head fwd")
            head_cpu_offload_handler.start_fwd()
            for idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids
                with head_cpu_offload_context if head_cpu_offload_context is not None else contextlib.nullcontext():
                    output = head_model(input_ids=input_ids, labels=labels)
                    head_cpu_offload_handler.on_minibatch_commit_forward()
                loss = output.loss
                head_losses.append(loss)
                if idx == mini_batch_num - 1:
                    break
            # tail fwd
            # time.sleep(0.5)
            print("tail fwd")
            for idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids
                output = tail_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                tail_losses.append(loss)
                # tail bwd
                tail_losses[idx].backward()
                if idx == mini_batch_num - 1:
                    break
            tail_optimizer.step()
            tail_optimizer.zero_grad()  # 优化器梯度归零
            # head bwd,simulate time consuming
            # time.sleep(0.5)
            print("head bwd")
            head_cpu_offload_handler.start_bwd()
            for j in range(mini_batch_num):
                head_cpu_offload_handler.on_minibatch_commit_backward()
                head_losses[j].backward()
            head_optimizer.step()
            head_optimizer.zero_grad()  # 优化器梯度归零
            prof.step()
            torch.cuda.reset_peak_memory_stats()


def test_model_param_offload():
    head_model, tail_model, head_optimizer, tail_optimizer, tokenizer, dataloader = _init()
    load_stream = torch.cuda.Stream(device)
    offload_stream = torch.cuda.Stream(device)
    head_p_off = ModelParamOffload(head_model, load_stream=load_stream, offload_stream=offload_stream)
    tail_p_off = ModelParamOffload(tail_model, load_stream=load_stream, offload_stream=offload_stream)
    head_os_off = OptimizerStateOffload(head_optimizer, load_stream=load_stream, offload_stream=offload_stream)
    tail_os_off = OptimizerStateOffload(tail_optimizer, load_stream=load_stream, offload_stream=offload_stream)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=0),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
        on_trace_ready=(torch.profiler.tensorboard_trace_handler("./log/trace", worker_name="param_offload")),  # 保存到 TensorBoard
        # on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            head_loss = 0
            head_losses = []
            tail_losses = []
            print("total iter------", i + 1)
            # head fwd
            print("head fwd")
            for idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids
                output = head_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                head_loss += loss.item()
                head_losses.append(loss)
                if idx == mini_batch_num - 1:
                    break
            tail_p_off.reload(True)
            tail_os_off.reload(True)
            head_p_off.offload(True)
            head_os_off.offload(True)
            head_p_off.wait_offload()
            head_os_off.wait_offload()
            tail_p_off.wait_reload()
            # tail fwd
            for idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids
                output = tail_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                tail_losses.append(loss)
                tail_losses[idx].backward()
                if idx == mini_batch_num - 1:
                    break
                # 每次梯度更新前裁剪梯度
            tail_os_off.wait_reload()
            tail_optimizer.step()
            tail_optimizer.zero_grad()  # 优化器梯度归零
            # head bwd,simulate time consuming
            head_p_off.reload(True)
            head_os_off.reload(True)
            tail_p_off.offload(True)
            tail_os_off.offload(True)
            head_p_off.wait_reload()
            tail_p_off.wait_offload()
            tail_os_off.wait_offload()
            # head_p_off.wait_offload()
            # time.sleep(0.5)
            for j in range(mini_batch_num):
                head_losses[j].backward()
            head_os_off.wait_reload()
            head_optimizer.step()
            head_optimizer.zero_grad()  # 优化器梯度归零
            prof.step()
            torch.cuda.reset_peak_memory_stats()
            print(f'finished ,loss : {head_loss}')


def test_all():
    head_model, tail_model, head_optimizer, tail_optimizer, tokenizer, dataloader = _init()
    load_stream = torch.cuda.Stream(device)
    offload_stream = torch.cuda.Stream(device)
    head_p_off = ModelParamOffload(head_model, load_stream=load_stream, offload_stream=offload_stream)
    tail_p_off = ModelParamOffload(tail_model, load_stream=load_stream, offload_stream=offload_stream)
    head_os_off = OptimizerStateOffload(head_optimizer, load_stream=load_stream, offload_stream=offload_stream)
    tail_os_off = OptimizerStateOffload(tail_optimizer, load_stream=load_stream, offload_stream=offload_stream)
    head_cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_minibatch=mini_batch_num, load_stream=load_stream, offload_stream=offload_stream
    )
    head_cpu_offload_context = CpuOffloadHookWithOffloadHandler(head_cpu_offload_handler)
    # tail_cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
    #     num_minibatch=mini_batch_num, load_stream=load_stream, offload_stream=offload_stream
    # )
    # tail_cpu_offload_context = CpuOffloadHookWithOffloadHandler(tail_cpu_offload_handler)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=0),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
        on_trace_ready=(torch.profiler.tensorboard_trace_handler("./log/trace", worker_name="all_offload")),  # 保存到 TensorBoard
        # on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            head_losses = []
            tail_losses = []
            print("total iter------", i + 1)
            # head fwd
            print("head fwd")
            head_cpu_offload_handler.start_fwd()
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels = input_ids
                with head_cpu_offload_context if head_cpu_offload_context is not None else contextlib.nullcontext():
                    output = head_model(input_ids=input_ids, labels=labels)
                    head_cpu_offload_handler.on_minibatch_commit_forward()
                loss = output.loss
                head_losses.append(loss)
            print(f'head fwd finished ,loss : {sum(head_losses) / len(head_losses)}')
            head_p_off.offload(False)
            head_os_off.offload(False)
            tail_p_off.reload(False)
            tail_os_off.reload(False)
            # head_p_off.offload(True)
            # head_os_off.offload(True)
            # tail_p_off.reload(True)
            # tail_os_off.reload(True)
            # head_p_off.wait_offload()
            # head_os_off.wait_offload()
            # tail_p_off.wait_reload()
            # tail_os_off.wait_reload()
            # tail fwd
            # time.sleep(0.5)
            print("tail fwd")
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels = input_ids
                output = tail_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                tail_losses.append(loss)
                # tail bwd
                # print("tail bwd")
                # for j in range(mini_batch_num):
                tail_losses[j].backward()
            tail_optimizer.step()
            tail_optimizer.zero_grad()  # 优化器梯度归零
            # head bwd,simulate time consuming
            tail_p_off.offload()
            tail_os_off.offload()
            head_p_off.reload()
            head_os_off.reload()
            # tail_p_off.offload(True)
            # tail_os_off.offload(True)
            # head_p_off.reload(True)
            # head_os_off.reload(True)
            # tail_p_off.wait_offload()
            # tail_os_off.wait_offload()
            # head_p_off.wait_reload()
            # head_os_off.wait_reload()
            # time.sleep(0.5)
            print("head bwd")
            head_cpu_offload_handler.start_bwd()
            for j in range(mini_batch_num):
                head_cpu_offload_handler.on_minibatch_commit_backward()
                head_losses[j].backward()
            head_optimizer.step()
            head_optimizer.zero_grad()  # 优化器梯度归零
            # prof.step()
            torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    # a = input("请输入测试模式：1-no_offload, 2-activation_offload, 3-param_offload, 4-all_offload:")
    # if a == "1":
    #     test_no_offload()
    # elif a == "2":
    #     test_activation_offload()
    # elif a == "3":
    #     test_model_param_offload()
    # elif a == "4":
    #     test_all()
    # else:
    #     print("输入错误，请重新输入！")
    test_model_param_offload()
