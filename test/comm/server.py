import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from usl.socket.socket_comm import SocketCommunicator


def send_tensor(tensor: torch.Tensor, dst: int, max_dim: int = 5):
    """
    用 NCCL 两次通信发送任意形状张量：
    1) padded shape（长度固定为 max_dim）
    2) tensor 本体（flatten 后）

    参数:
        tensor (Tensor): 要发送的张量（GPU 上）
        dst (int): 目标 rank
        max_dim (int): 最大维度数
    """
    assert tensor.is_cuda, "Tensor 必须在 GPU 上（NCCL 一定要在 GPU）"

    # ----- 1) padded shape -----
    shape = list(tensor.shape)
    padded_shape = shape + [0] * (max_dim - len(shape))
    shape_tensor = torch.tensor(padded_shape, dtype=torch.int64, device=tensor.device)

    # broadcast shape first
    dist.send(shape_tensor, dst=dst)

    # ----- 2) send flattened tensor -----
    flat_tensor = tensor.contiguous().view(-1)
    dist.send(flat_tensor, dst=dst)


def recv_tensor(src: int, dtype: torch.dtype, max_dim: int = 5):
    """
    接收与上面 send_tensor 对应的函数。

    返回一个恢复形状后的张量。
    """
    device = torch.cuda.current_device()
    # ----- 1) 接收 padded shape -----
    shape_tensor = torch.empty(max_dim, dtype=torch.int64, device=device)
    dist.recv(shape_tensor, src=src)

    # 去掉 padding（即 shape 中的 0）
    real_shape = [int(s) for s in shape_tensor.tolist() if s != 0]

    # ----- 2) 接收 flat tensor -----
    numel = int(torch.prod(torch.tensor(real_shape)))
    flat_tensor = torch.empty(numel, dtype=dtype, device=device)
    dist.recv(flat_tensor, src=src)

    # reshape 回原 tensor
    return flat_tensor.view(*real_shape)


def train(rank, world_size):
    # 每个进程使用不同的 GPU
    torch.cuda.set_device(rank)
    print(f'Rank {rank} started training on GPU {rank}')
    cur_phase = 'FWD'
    global_batch_size = 4
    # init communicator
    if rank == 0 or rank == world_size - 1:
        print(f'Rank {rank} is waiting for the client to connect')
        communicator = SocketCommunicator(
            is_server=True, host='localhost', port=9000 if rank == 0 else 9001, buffer_size=1024 * 4, rate_limit_mbps=230
        )
        communicator.accept_client()
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    # do recv grad or acti
    # do server fwd F than B
    cur_mb_idx = 0
    while cur_mb_idx < global_batch_size:
        if rank == 0:
            print('Rank 0 is receiving activation from client')
            t: torch.Tensor = communicator.receive()
            t = t.cuda(rank)
        else:
            t = recv_tensor(src=rank - 1, dtype=torch.float32, max_dim=5)
        # do training
        t.add_(1.0)
        print(rank, 'fwd activation: ', t[0, :5, :5])
        if rank < world_size - 1:
            # dist.send(tensor=t, dst=rank + 1)
            send_tensor(tensor=t, dst=rank + 1)
        else:
            communicator.send(t.cpu())
        # server backward
        if rank == world_size - 1:
            t: torch.Tensor = communicator.receive()
            t = t.cuda(rank)
            print('Rank 3 is receiving gradient from client')
        else:
            t = recv_tensor(src=rank + 1, dtype=torch.float32, max_dim=5)
        print(rank, 'bwd gradient: ', t[0, :5, :5])
        # server send grad to client
        if rank == 0:
            print('Rank 0 is sending gradient to client')
            communicator.send(t.cpu())
        else:
            send_tensor(tensor=t, dst=rank - 1)
        # do training
        cur_mb_idx += 1
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 or rank == world_size - 1:
        communicator.close()
    # 模型和数据处理的代码


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
