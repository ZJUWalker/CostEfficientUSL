import os
import time
import torch
import argparse
from transformers import AutoConfig
from usl.server.server_base import ServerArgs
from usl.server.single_server import SingleServer
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
    server_model = load_server_model(model_dir, server_args.model, split_point, use_lora=True)  # use_lora=True for LoRA
    # =====================================================================
    torch.cuda.init()
    torch.cuda.set_device(server_args.server_device)
    torch.cuda.reset_peak_memory_stats()
    server = SingleServer(server_args, server_model, logger=logger, matrix_logger=matrix_logger)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8000, help="Port to listen")
    parser.add_argument("-S", "--step", type=int, default=5, help="Number of steps to profile")
    parser.add_argument("-L", "--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="Model card")
    parser.add_argument("-SD", "--server_device", type=str, default="cuda:0", help="Device for server model")
    parser.add_argument("-SP", "--split_point", type=int, default=2)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mbps", type=int, default=100)
    args = parser.parse_args()
    args = ServerArgs(
        port=args.port,
        step=args.step,
        use_lora=args.use_lora,
        model=args.model,
        server_device=args.server_device,
        split_point=args.split_point,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        rete_limit_mbps=args.mbps,
    )

    set_seed(SEED)
    run_server(args)
