import os
import time
import torch
import argparse
from transformers import AutoConfig
from usl.server.single_server import SingleServer, ServerArgs, convert_pipeline_mode
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
    parser.add_argument("-SD", "--server_device", type=str, default="cuda:1", help="Device for server model")
    parser.add_argument("-SP", "--split_point", type=int, default=4)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mbps", type=int, default=0)
    parser.add_argument("--pmode", type=str, default="gpipe", help='mode of pipeline, "strict" or "loose" or "1f1b"')
    parser.add_argument("--offload_activation", "-OA", action="store_true", default=False)
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    args = parser.parse_args()
    args = ServerArgs(
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
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
    )
    print(args)

    set_seed(SEED)
    run_server(args)
