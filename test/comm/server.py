import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from usl.socket.socket_comm import SocketCommunicator


def train(rank, world_size):
    # 每个进程使用不同的 GPU
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(f'Rank {rank} started training on GPU {rank}')
    if rank == 0 or rank == world_size - 1:
        print(f'Rank {rank} is waiting for the client to connect')
        communicator = SocketCommunicator(
            is_server=True, host='localhost', port=8000 if rank == 0 else 8001, buffer_size=1024 * 4, rate_limit_mbps=230
        )
        communicator.accept_client()
    # do recv grad or acti
    if rank == 0:
        t: torch.Tensor = communicator.receive()
        t = t.cuda(rank)
    else:
        t = torch.rand(1000, 1000).cuda(rank)
        dist.recv(tensor=t, src=rank - 1)
    # do training
    t.add_(1.0)
    print(rank, t[:5, :5])
    # do send grad or acti
    if rank < world_size - 1:
        dist.send(tensor=t, dst=rank + 1)
    else:
        communicator.send(t.cpu())
    dist.barrier()

    dist.destroy_process_group()
    # 模型和数据处理的代码


def main():
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    main()
