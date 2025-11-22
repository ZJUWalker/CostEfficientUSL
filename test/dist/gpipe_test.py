import logging
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, PipelineStage
from usl.server.base import ServerArgs
from usl.utils.dataset.exp import AverageMeter
from usl.utils.load_utils import *
import torch.multiprocessing as mp
from transformers import AutoTokenizer


def init_logging(rank):
    import logging
    import os

    # 最详细分布式 log
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"pipeline_rank{rank}.log", mode="a"), logging.StreamHandler()],
    )


def run_gpipe(rank, world_size, stage, optimizer, mb_num, dataloader, loss_fn, log_step, device, print_grad=False):
    init_logging(rank)
    avg_loss = AverageMeter()
    losses = []
    schedule = ScheduleGPipe(stage, mb_num, loss_fn=loss_fn)
    print(f"Rank {rank} start training...")
    # Train the model
    for idx, batch in enumerate(dataloader, 1):
        if rank == 0 or rank == world_size - 1:
            # input_ids = batch["input_ids"].to(device)
            input_hidden_state = torch.randn(8, 100, 2048).to(device)
            # attention_mask = torch.zeros(8, 1, 100, 100).to(device)
            # x, target = batch[0].to(device), batch[1].to(device)
        if rank == 0:
            # Input data
            schedule.step(input_hidden_state)
        elif rank == world_size - 1:
            target = torch.randn(8, 100, 2048).to(device)
            output = schedule.step(target=target, losses=losses)
            avg_loss.update(torch.stack(losses).mean().item())
            # Update the model
            if idx % log_step == 0 or idx == len(dataloader):
                print(f"Step ({idx-log_step},{idx}): Average loss: {avg_loss.avg}")
                avg_loss.reset()
                losses.clear()
        else:
            schedule.step()
        optimizer.step()
        optimizer.zero_grad()
        # global_step += 1

    pass


def run(rank, world_size, server_args: ServerArgs, type=0):
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(rank=rank, world_size=world_size)
    dist_group = dist.new_group(ranks=list(range(world_size)))
    model_dir = os.path.join("data/models", server_args.model)
    split_point = server_args.split_point
    device = f'cuda:{rank}'
    dataset_name = server_args.dataset
    batch_size = server_args.batch_size
    model_name = server_args.model
    max_seq_len = 512
    log_step = 1
    mb_num = server_args.batch_size // server_args.micro_batch_size
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    server_model = load_server_model(
        model_dir,
        model_name,
        split_point,
        use_lora=server_args.use_lora,
    )
    model = manual_model_split(server_model, rank, world_size, device)
    stage = PipelineStage(
        model,
        rank,
        world_size,
        device,
        # input_args=(torch.randn(1, 100, 2048, device='meta'), torch.zeros(1, 1, 100, 100, device='meta')),
        # output_args=(torch.randn(1, 100, 2048, device='meta'),),
    )
    # print(f"Rank {rank} model: {stage.submod}")
    # Create an optimizer
    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-3)
    # Define a loss function
    # loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    loss_fn = lambda x, y: torch.sub(x, y).abs().mean()
    # Create a dataloader
    client_dataloaders = load_dataset(dataset_name, tokenizer, [0], batch_size, max_seq_len)
    dataloader = client_dataloaders[0]['train']  # 默认只取第一个客户端数据
    # run gpipe
    # if type == 0:
    # print(stage.__class__.__name__)
    run_gpipe(rank, world_size, stage, optimizer, mb_num, dataloader, loss_fn, log_step, device)
    # # run manual gpipe
    # elif type == 1:
    #     run_manual_gpipe(rank, world_size, stage, optimizer, mb_num, dataloader, loss_fn, batch_size, log_step, device)
    # elif type == 2:
    #     run_split_server_gpipe(rank, world_size, stage, optimizer, mb_num, dataloader, batch_size, log_step, device)
    dist.destroy_process_group()


'''
Args:
    --type: 0 for gpipe, 1 for manual gpipe,2 for u-shape split server gpipe
'''
if __name__ == "__main__":
    import argparse

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '4'
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8888, help="Port to listen")
    parser.add_argument("-S", "--step", type=int, default=5, help="Number of steps to profile")
    parser.add_argument("-L", "--lora", action="store_true", help="Use LoRA")
    parser.add_argument("-M", "--model", type=str, default="qwen/qwen3-1.7b", help="Model card")
    parser.add_argument("-SD", "--server_device", type=str, default="cuda:2", help="Device for server model")
    parser.add_argument("-SP", "--split_point", type=int, default=4)
    parser.add_argument("-DS", "--dataset", type=str, default="dialogsum")
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mbps", type=int, default=230)
    parser.add_argument("--pmode", type=str, default="pdwc", help='mode of pipeline, "strict" or "loose" or "1f1b"')
    parser.add_argument("--offload_activation", "-OA", action="store_true")
    parser.add_argument("--offload_activation_mb_num", "-OAM", type=int, default=0)
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--prof", action="store_true")
    parser.add_argument('--type', type=int, default=0)
    args = parser.parse_args()
    server_args = ServerArgs(
        port=args.port,
        step=args.step,
        use_lora=args.lora,
        model=args.model,
        server_device=args.server_device,
        split_point=args.split_point,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        rate_limit_mbps=args.mbps,
        pipeline_mode='gpipe',
        offload_activation=args.offload_activation,
        offload_activation_mb_num=args.offload_activation_mb_num,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        prof=args.prof,
    )
    world_size = int(os.environ['WORLD_SIZE'])
    mp.spawn(
        run,
        args=(
            world_size,
            server_args,
            args.type,
        ),
        nprocs=world_size,
        join=True,
    )
