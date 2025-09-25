import torch
import torch.nn as nn
from usl.offload import ActivationOffload


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def _get_memory_usage_mb():
    return round(torch.cuda.memory_allocated() / 1024**2, 4)


def test_activation_offload():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    collector = ActivationOffload(model, offload_threshold=0, device=device)
    print('init', _get_memory_usage_mb(), 'MB')
    for i in range(5):
        print('step-------------------', i, '----------------')
        print('before forward', _get_memory_usage_mb(), 'MB')
        input = torch.randn(1000, 500).to(device)
        output = model(input)
        loss = output.sum()
        print('before offload', _get_memory_usage_mb(), 'MB')
        collector.offload(async_offload=True)
        collector.wait_offload()
        print('after offload', _get_memory_usage_mb(), 'MB')
        collector.reload(async_reload=True)
        collector.wait_reload()
        print('after reload', _get_memory_usage_mb(), 'MB')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        collector.clear()
        print('after step', _get_memory_usage_mb(), 'MB')


test_activation_offload()
