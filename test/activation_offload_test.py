import torch
import torch.nn as nn
from usl.offload.async_activation_offload import (
    AsyncDoubleBufferGroupOffloadHandler,
    CpuOffloadHookWithOffloadHandler,
    SynchronizedGroupOffloadHandler,
)
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import contextlib


def _get_memory_usage_mb():
    return round(torch.cuda.memory_allocated() / 1024**2, 4), round(torch.cuda.max_memory_allocated() / 1024**2, 4)


def test_activation_offload():
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.Stream(device))
    model = GPT2LMHeadModel(GPT2Config(n_embd=1280, n_head=20, n_inner=4096, n_layer=4)).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(num_minibatch=4) # 异步的卸载
    cpu_offload_handler = SynchronizedGroupOffloadHandler(num_minibatch=4) # 同步的卸载
    cpu_offload_context = CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler)

    print("init", _get_memory_usage_mb(), "MB")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1, warmup=2, active=2, repeat=0
        ),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
        on_trace_ready=(
            torch.profiler.tensorboard_trace_handler("./log/trace", worker_name="activation_offload")
        ),  # 保存到 TensorBoard
        # on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            losses = []
            print("iter------", i + 1)
            cpu_offload_handler.start_fwd()
            for j in range(4):
                input_ids = torch.randint(0, 10000, (4, 512), device=device)
                # attention_mask = torch.ones((4, 512), device=device)s
                labels = input_ids
                with cpu_offload_context if cpu_offload_context is not None else contextlib.nullcontext():
                    #     #     # print('before forward', _get_memory_usage_mb(), 'MB')
                    output = model(input_ids=input_ids, labels=labels)
                    cpu_offload_handler.on_minibatch_commit_forward()
                # torch.cuda.synchronize()
                # print(f"step {i+1}, mini batch {j+1}, fwd done,mem usage", _get_memory_usage_mb(), "MB")
                loss = output.loss
                losses.append(loss)
            cpu_offload_handler.start_bwd()
            for j in range(4):
                # print(f"step {i+1}, mini batch {j}, before backward,mem usage", _get_memory_usage_mb(), "MB")
                cpu_offload_handler.on_minibatch_commit_backward()
                losses[j].backward()
            print(f"step {i+1} loss: {sum(losses)/4:.6f},mem usage", _get_memory_usage_mb(), "MB")
            optimizer.step()
            optimizer.zero_grad()
            prof.step()
            # print(f"step {i+1} optimizer done,mem usage", _get_memory_usage_mb(), "MB")
            torch.cuda.reset_peak_memory_stats()


test_activation_offload()
