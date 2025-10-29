import os
import time
import torch
import argparse
from transformers import AutoConfig
from usl.server.single_server import PipelineMode, SingleServer, ServerArgs, convert_pipeline_mode
from usl.utils.exp import set_seed
from usl.utils.load_utils import *
from usl.utils.log_utils import create_logger
from usl.server import *

SEED = 0


def run_server(server_args: ServerArgs):
    # =====================================================================
    log_dir = f"log/{server_args.model}/server"
    logger = create_logger(log_file_name="training_steps.log", log_dir=log_dir, console_output=False)
    matrix_logger = create_logger(
        log_file_name="training_metrics.log",
        log_dir=log_dir,
        console_output=False,
    )
    matrix_logger.info(f"step | mem alloc(GB) | mem reserved(GB) | avg_loss ")
    # =====================================================================
    model_dir = os.path.join("data/models", server_args.model)
    split_point = server_args.split_point
    server_model = load_server_model(model_dir, server_args.model, split_point, use_lora=server_args.use_lora)  # use_lora=True for LoRA
    # =====================================================================
    torch.cuda.init()
    torch.cuda.set_device(server_args.server_device)
    torch.cuda.reset_peak_memory_stats()
    server = SingleServer(server_args, server_model, logger=logger, matrix_logger=matrix_logger)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8888, help="Port to listen")
    parser.add_argument("-S", "--step", type=int, default=5, help="Number of steps to profile")
    parser.add_argument("-L", "--lora", action="store_true", help="Use LoRA")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="Model card")
    parser.add_argument("-SD", "--server_device", type=str, default="cuda:2", help="Device for server model")
    parser.add_argument("-SP", "--split_point", type=int, default=4)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mbps", type=int, default=300)
    parser.add_argument("--pmode", type=str, default="pdwc", help='mode of pipeline, "strict" or "loose" or "1f1b"')
    parser.add_argument("--offload_activation", "-OA", action="store_true")
    parser.add_argument("--offload_activation_mb_num", "-OAM", type=int, default=0)
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--prof", action="store_true")
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
        pipeline_mode=convert_pipeline_mode(args.pmode),
        offload_activation=args.offload_activation,
        offload_activation_mb_num=args.offload_activation_mb_num,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        prof=args.prof,
    )
    # 只要看到offload_activation_mb_num大于0，就默认开启offload_activation
    # 如果offload_activation, 则offload_activation_mb_num=batch_size/micro_batch_size
    if server_args.offload_activation or server_args.offload_activation_mb_num > 0:
        server_args.offload_activation_mb_num = max(
            0, min(server_args.offload_activation_mb_num)
        )
        server_args.offload_activation = True
    else:
        server_args.offload_activation_mb_num = 0
        server_args.offload_activation = False
    # print(args)
    if server_args.offload_activation and server_args.pipeline_mode != PipelineMode.PIPE_DREAM_WC:
        print("Warning!Offload activation is only supported in pipedream_wc mode, or else it will not be effective.")
    set_seed(SEED)
    run_server(server_args)
