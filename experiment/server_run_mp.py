import os
import torch
import torch.distributed as dist

from usl.server.pipeline import ServerScheduleGPipe, ServerSchedule1F1B, ServerPipelineStage, ServerPipelineScheduleSingle
from usl.server.base import ServerArgs, PipelineMode, convert_pipeline_mode
from usl.utils.dataset.exp import AverageMeter
from usl.utils.exp import set_seed
from usl.utils.load_utils import *
import torch.multiprocessing as mp
from transformers import AutoTokenizer


def run_pipeline(
    rank: int, world_size: int, scheduler: ServerPipelineScheduleSingle, optimizer: torch.optim.Optimizer, mb_num: int, epoch: int = 1, step: int = 5, profile=False
):
    stage = scheduler._stage
    # schedule = ServerScheduleGPipe(stage, mb_num)  # don't need loss_fn
    print(f"Rank {rank} start {scheduler.__class__.__name__} training...,num_microbatches={mb_num},is first={stage.is_first},is last={stage.is_last}")
    # Train the model
    total_steps = epoch * step
    curr_step = 0
    save_steps = 200
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=0),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
        # on_trace_ready=(
        #     torch.profiler.tensorboard_trace_handler("./log/trace", worker_name=f"gpipe_ws_{stage.group_size}") if rank == 0 and profile else None
        # ),  # 保存到 TensorBoard
        on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        while curr_step < total_steps:
            if rank == 0:
                print(f"Server globle step {curr_step} start...")
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            curr_step += 1
            
            # ================= [新增：定期保存 Checkpoint] =================
            # if curr_step % save_steps == 0 or curr_step == step:
            #     # 注意等待当前 step 的多卡同步完成再保存
            #     if world_size > 1:
            #         dist.barrier()
                    
            #     ckpt_dir = os.path.join(f"data/save_models/server/trunk_rank_{rank}", f"checkpoint-{curr_step}")
            #     os.makedirs(ckpt_dir, exist_ok=True)
                
            #     if rank == 0:
            #         print(f"Server saving checkpoint at step {curr_step} to {ckpt_dir}...")
                
            #     model_to_save = stage.submod.module if hasattr(stage.submod, "module") else stage.submod
            #     if hasattr(model_to_save, "save_pretrained"):
            #         model_to_save.save_pretrained(ckpt_dir)
            #     else:
            #         torch.save(model_to_save.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
            # ===============================================================
            
            if profile and rank == 0:
                print(f"prof step")
                prof.step()
    if world_size > 1:
        dist.barrier()
    scheduler.send_profile_res()
    pass


def run(rank, world_size, server_args: ServerArgs):
    set_seed(0)
    dist.init_process_group(rank=rank, world_size=world_size)
    model_dir = os.path.join("data/models", server_args.model)
    split_point = server_args.split_point
    server_args.server_device = f'cuda:{rank+4}'
    device = f'cuda:{rank+4}'
    torch.cuda.set_device(device)
    model_name = server_args.model
    max_seq_len = 512
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
    stage = ServerPipelineStage(
        model,
        rank,
        world_size,
        device,
        input_args=(
            torch.randn(server_args.micro_batch_size, max_seq_len, model.config.hidden_size, device='meta'),
            torch.zeros(server_args.micro_batch_size, 1, max_seq_len, max_seq_len, device='meta'),
        ),
        mbps_limit=server_args.rate_limit_mbps,
        port=server_args.port,
        use_qlora_comm=server_args.use_qlora_comm,
    )
    stage._init_p2p_neighbors()  # check connection
    # print(f"Rank {rank} model: {stage.submod}")
    # Create an optimizer
    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-3)
    if server_args.pipeline_mode in [PipelineMode.GPIPE, PipelineMode.PIPE_DREAM_WC, PipelineMode.NAIVE]:
        print('offload_activation_mb_num:', server_args.offload_activation_mb_num)
        scheduler = ServerScheduleGPipe(stage, mb_num, offload_activation_mb_num=server_args.offload_activation_mb_num)  # don't need loss_fn
    elif server_args.pipeline_mode == PipelineMode.PIPE_DREAM_STRICT:
        scheduler = ServerSchedule1F1B(stage, mb_num)  # don't need loss_fn
    else:
        raise NotImplementedError('other pipeline methods are not implemeneted yet')
    run_pipeline(rank, world_size, scheduler, optimizer, mb_num, epoch=server_args.epoch, step=server_args.step, profile=server_args.prof)
    dist.destroy_process_group()


'''
Args:
    --type: 0 for gpipe, 1 for manual gpipe,2 for u-shape split server gpipe
'''
if __name__ == "__main__":
    import argparse

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8888, help="Port to listen")
    parser.add_argument("-S", "--step", type=int, default=20, help="Number of steps to profile")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs to profile")
    parser.add_argument("-L", "--lora", action="store_true", help="Use LoRA")
    parser.add_argument("-M", "--model", type=str, default="qwen/qwen3-1.7b", help="Model card")
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
    # parser.add_argument('--type', type=int, default=0)?
    parser.add_argument('--world_size', '-WS', type=int, default=4)
    parser.add_argument('--qloracomm','-Q',action='store_true', default=False, help='Whether to use QLoRA compression for communication.')
    args = parser.parse_args()
    server_args = ServerArgs(
        port=args.port,
        step=args.step,
        epoch=args.epoch,
        use_lora=args.lora,
        model=args.model,
        split_point=args.split_point,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        rate_limit_mbps=args.mbps,
        pipeline_mode=convert_pipeline_mode(args.pmode),
        offload_activation=args.offload_activation,
        offload_activation_mb_num=args.offload_activation_mb_num,
        batch_size=args.batch_size,
        world_size=args.world_size,
        micro_batch_size=args.micro_batch_size,
        prof=args.prof,
        use_qlora_comm=args.qloracomm,
    )
    os.environ['WORLD_SIZE'] = str(args.world_size)
    world_size = int(os.environ['WORLD_SIZE'])
    # post process offload args
    if server_args.offload_activation:
        server_args.offload_activation_mb_num = server_args.batch_size // server_args.micro_batch_size
    elif server_args.offload_activation_mb_num > 0:
        server_args.offload_activation = True
        server_args.offload_activation_mb_num = min(server_args.offload_activation_mb_num, server_args.batch_size // server_args.micro_batch_size)
    else:
        server_args.offload_activation_mb_num = 0
        server_args.offload_activation = False
    # print(args)
    if server_args.offload_activation and server_args.pipeline_mode != PipelineMode.PIPE_DREAM_WC:
        print("Warning!Offload activation is only supported in pipedream_wc mode, or else it will not be effective.")
    mp.spawn(
        run,
        args=(
            world_size,
            server_args,
        ),
        nprocs=world_size,
        join=True,
    )
