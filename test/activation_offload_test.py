import torch
import torch.nn as nn
from usl.offload import ActivationOffload
from usl.offload.activation_offload import ActivationOffload
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(5000, 5000)
#         self.fc2 = nn.Linear(5000, 5000)
#         self.fc3 = nn.Linear(5000, 100)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x


def _get_memory_usage_mb():
    return round(torch.cuda.memory_allocated() / 1024**2, 4), round(torch.cuda.max_memory_allocated() / 1024**2, 4)


def test_activation_offload():
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.Stream(device))
    offload_stream = torch.cuda.Stream(device)
    load_stream = torch.cuda.Stream(device)
    model = GPT2LMHeadModel(GPT2Config(n_embd=1024, n_head=16, n_inner=4096, n_layer=4)).to(device)
    model.train()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    offloader = ActivationOffload(
        device=device,  # 或 None
        pin_memory=True,  # 建议 True
        threshold_bytes=1024,
        async_offload=True,
        offload_stream=offload_stream,
        reload_stream=load_stream,
    )
    print('init', _get_memory_usage_mb(), 'MB')
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
            print('iter------', i + 1)
            input_ids = torch.randint(0, 10000, (4, 512), device=device)
            # attention_mask = torch.ones((4, 512), device=device)s
            labels = input_ids
            # print('before forward', _get_memory_usage_mb(), 'MB')
            with offloader:
                output = model(input_ids=input_ids, labels=labels)
                loss = output.loss
                loss.backward()
                print(f'step {i+1} loss: {loss.item()},mem usage', _get_memory_usage_mb(), 'MB')
            prof.step()
            torch.cuda.reset_peak_memory_stats()


test_activation_offload()
