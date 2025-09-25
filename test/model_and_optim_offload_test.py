import torch
import torch.nn as nn
from usl.offload import ActivationOffload, ModelParamOffload, OptimizerStateOffload


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x


def _get_memory_usage_mb():
    return round(torch.cuda.memory_allocated() / 1024**2, 4)


def test_model_offload():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device)
    torch.cuda.set_stream(stream)
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_mgr = ModelParamOffload(model, offload_threshold=0, device=device)
    optimizer_mgr = OptimizerStateOffload(optimizer, offload_threshold=0, device=device)
    print('init', _get_memory_usage_mb(), 'MB')
    for i in range(5):
        print('step-------------------', i, '----------------')
        print('before forward', _get_memory_usage_mb(), 'MB')
        input = torch.randn(1, 500).to(device)
        model_mgr.reload()
        print('after load model', _get_memory_usage_mb(), 'MB')
        output = model(input)
        loss = output.sum()
        print('before backward', _get_memory_usage_mb(), 'MB')
        loss.backward()
        print('after backward', _get_memory_usage_mb(), 'MB')
        optimizer_mgr.reload()
        print('after load optimizer', _get_memory_usage_mb(), 'MB')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print('after step', _get_memory_usage_mb(), 'MB')
        optimizer_mgr.offload()
        print('after offload optimizer', _get_memory_usage_mb(), 'MB')
        model_mgr.offload()
        print('after offload model', _get_memory_usage_mb(), 'MB')


test_model_offload()
