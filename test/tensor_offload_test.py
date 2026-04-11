import os
import time
from typing import Any, Tuple
import torch
import torch.nn as nn
from usl.offload.activation_offload import (
    AsyncDoubleBufferGroupOffloadHandler,
    CpuOffloadHookWithOffloadHandler,
    SynchronizedGroupOffloadHandler,
)
from usl.offload import ModelParamOffload, OptimizerStateOffload, AsyncModelParamOffloadHandler
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import contextlib
from usl.utils.load_utils import load_dataset, load_client

mini_batch_num = 8
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


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
    model_name = 'qwen/qwen3-8b'  # 'qwen/qwen3-1.7b'
    split_point = 2
    model_dir = os.path.join("data/models", model_name)
    head_model, tail_model, tokenizer = load_client(model_dir, model_name, split_point, use_lora=True, use_qlora_4bit=False, use_qlora_8bit=False)
    # tail_model.train()
    head_model.to(device)
    print("load head", _get_memory_usage_mb(), "MB")
    # tail_model.to(device)
    # print('load tail', _get_memory_usage_mb(), 'MB')
    head_optimizer = torch.optim.Adam(head_model.parameters(), lr=0.001)
    tail_optimizer = torch.optim.Adam(tail_model.parameters(), lr=0.001)
    return head_model, tail_model, head_optimizer, tail_optimizer


def test_no_offload():
    head_model, tail_model, head_optimizer, tail_optimizer = _init()
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
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels = input_ids
                output = head_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                head_losses.append(loss)
            # tail fwd
            # time.sleep(0.5)
            print("tail fwd")
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels = input_ids
                output = tail_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                tail_losses.append(loss)
                tail_losses[j].backward()
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
    head_model, tail_model, head_optimizer, tail_optimizer = _init()
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
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels = input_ids
                with head_cpu_offload_context if head_cpu_offload_context is not None else contextlib.nullcontext():
                    output = head_model(input_ids=input_ids, labels=labels)
                    head_cpu_offload_handler.on_minibatch_commit_forward()
                loss = output.loss
                head_losses.append(loss)
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
                tail_losses[j].backward()
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
    head_model, tail_model, head_optimizer, tail_optimizer = _init()
    load_stream = torch.cuda.Stream(device)
    offload_stream = torch.cuda.Stream(device)
    except_tensor_list = [id(p) for p in tail_model.lm_head.parameters()]
    head_p_off = AsyncModelParamOffloadHandler(head_model, device, load_stream=load_stream, offload_stream=offload_stream)
    tail_p_off = AsyncModelParamOffloadHandler(tail_model, device, load_stream=load_stream, offload_stream=offload_stream)
    # except_tensor_list = []
    # head_p_off = ModelParamOffload(head_model, load_stream=load_stream, offload_stream=offload_stream, except_tensor_idx_list=except_tensor_list)
    # tail_p_off = ModelParamOffload(tail_model, load_stream=load_stream, offload_stream=offload_stream, except_tensor_idx_list=except_tensor_list)
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
            head_outputs = []
            tail_inputs = []
            labels = []
            tail_losses = []

            print("total iter------", i + 1)
            # head fwd
            print("head fwd")
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels.append(input_ids)
                with head_p_off:
                    output: Tuple[torch.Tensor, torch.Tensor, Any] = head_model(input_ids=input_ids)
                    head_outputs.append(output[0])
                tail_inputs.append(output[0].detach())
            print(f'curr mem usage before head offload: {_get_memory_usage_mb()} MB')
            head_p_off.offload()
            print(f'curr mem usage after head offload: {_get_memory_usage_mb()} MB')
            tail_p_off.reload()
            print(f'curr mem usage after tail reload: {_get_memory_usage_mb()} MB')
            # head_p_off.wait_offload()

            # tail_p_off.wait_reload()
            # print(f'curr mem usage after reload: {_get_memory_usage_mb()} MB')
            # for n, p in head_model.named_parameters():
            # print(n, p.shape, p.device)
            # tail fwd
            for j in range(mini_batch_num):
                label = labels[j]
                tail_inputs[j].requires_grad = True
                with tail_p_off:
                    output = tail_model(hidden_states=tail_inputs[j], labels=label)
                loss = output.loss
                tail_losses.append(loss)
                tail_losses[j].backward()
            # 每次梯度更新前裁剪梯度
            tail_optimizer.step()
            tail_optimizer.zero_grad()  # 优化器梯度归零
            # head bwd,simulate time consuming
            print(f'curr mem usage before head bwd: {_get_memory_usage_mb()} MB')
            tail_p_off.offload()
            print(f'curr mem usage after tail offload: {_get_memory_usage_mb()} MB')
            head_p_off.reload()
            print(f'curr mem usage after head reload: {_get_memory_usage_mb()} MB')
            # head_p_off.wait_reload()
            # tail_p_off.wait_offload()
            for j in range(mini_batch_num):
                head_outputs[j].backward(tail_inputs[j].grad)
            # wait for head_os_off to finish reload and offload
            head_optimizer.step()
            head_optimizer.zero_grad()  # 优化器梯度归零
            prof.step()
            torch.cuda.reset_peak_memory_stats()
            print(f'finished ,loss : {sum(tail_losses)}')
            head_p_off.update_param_ptr()
            # tail_p_off.update_param_ptr()

            # head_p_off.clear_buffer()


def test_all():
    head_model, tail_model, head_optimizer, tail_optimizer = _init()
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
            head_loss = 0
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
                head_loss += loss.item()
                head_losses.append(loss)
            tail_p_off.reload(True)
            tail_os_off.reload(True)
            head_p_off.offload(True)
            head_os_off.offload(True)
            head_p_off.wait_offload()
            head_os_off.wait_offload()
            tail_p_off.wait_reload()
            for j in range(mini_batch_num):
                input_ids = torch.randint(0, 10000, (1, 512), device=device)
                labels = input_ids
                output = tail_model(input_ids=input_ids, labels=labels)
                loss = output.loss
                tail_losses.append(loss)
                tail_losses[j].backward()
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
            head_cpu_offload_handler.start_bwd()
            for j in range(mini_batch_num):
                head_cpu_offload_handler.on_minibatch_commit_backward()
                head_losses[j].backward()
            head_os_off.wait_reload()
            head_optimizer.step()
            head_optimizer.zero_grad()  # 优化器梯度归零
            prof.step()
            torch.cuda.reset_peak_memory_stats()
            print(f'finished ,loss : {head_loss}')


if __name__ == '__main__':
    test_model_param_offload()
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
    # # test_model_param_offload()
