import os
import argparse
import warnings
import sys

import torch
from usl.client import ClientArgs
from usl.utils.log_utils import create_logger
from usl.utils.load_utils import load_client, load_dataset
from usl.utils.exp import set_seed
from usl.server.single_server import PipelineMode, convert_pipeline_mode
from usl.client import (
    GPipeClientTrainer,
    SequentialClientTrainer,
    PipeDreamStrictClientTrainer,
    PipeDreamWCClientTrainer,
    PipeDreamWCEagerClientTrainer,
)


SEED = 0
warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)


def run_client(args: ClientArgs, profile=False):
    set_seed(SEED)
    dataset_name = args.dataset
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    model_name = args.model
    model_dir = os.path.join("data/models", model_name)
    split_point = args.split_point
    lora = args.use_lora
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    log_dir = f"log/{args.model}/client"
    logger = create_logger(log_file_name="client.log", console_output=False, log_dir=log_dir)
    logger.info(f"client start with args: {args}")

    # ---------------load model and tokenizer --------------------------
    head, tail, tokenizer = load_client(
        model_dir, model_name, split_point, use_lora=(lora and split_point > 0), use_qlora_4bit=False, use_qlora_8bit=False
    )
    # ---------------load dataset------------------------------------
    client_dataloaders = load_dataset(dataset_name, tokenizer, [0], batch_size, max_seq_len)
    dataloader = client_dataloaders[0]  # 默认只取第一个客户端数据

    # -----------------create client----------
    if args.pipeline_mode == PipelineMode.GPIPE:
        Trainer = GPipeClientTrainer
    elif args.pipeline_mode == PipelineMode.PIPE_DREAM_STRICT:
        Trainer = PipeDreamStrictClientTrainer
    elif args.pipeline_mode == PipelineMode.PIPE_DREAM_WC:
        Trainer = PipeDreamWCClientTrainer
    elif args.pipeline_mode == PipelineMode.PIPE_DREAM_WC_EAGER:
        Trainer = PipeDreamWCEagerClientTrainer
    else:
        Trainer = SequentialClientTrainer  # default sequential trainer
    client = Trainer(
        client_args=args,
        head_model=head,
        tail_model=tail,
        tokenizer=tokenizer,
        client_device=device,
        train_logger=logger,
        dataset_train=dataloader["train"],
        dataset_test=dataloader["test"],
        num_workers=args.micro_batch_size,
    )
    client.train_epoch(profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8888, help="port to listen")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-SL", "--max_seq_len", type=int, default=512, help="max sequence length")
    parser.add_argument("-S", "--step", type=int, default=5)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-E", "--epoch", type=int, default=1)
    parser.add_argument("-SP", "--split_point", type=int, default=4)
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--mbps", type=int, default=300)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--offload_activation_mb_num", "-OAM", type=int, default=0)
    parser.add_argument("--offload_model_state_sp_num", "-OSSP", type=int, default=0)
    parser.add_argument("--offload_activation", "-OA", action="store_true", default=False)
    parser.add_argument("--offload_model_state", "-OS", action="store_true", default=False)
    parser.add_argument("--sort", type=str, default="no", help='sort batch before pipeline, "no" or "desc" or "asc"')
    parser.add_argument("--pmode", type=str, default="pdwc", help='pipeline mode, "strict" or "wc" or "eager"')
    parser.add_argument("--profile", "-PROF", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="log/profile")
    parser.add_argument('--max_client_mem_gb', type=int, default=24, help='The maximum memory allocation for the client.')
    args = parser.parse_args()
    profile = args.profile

    args = ClientArgs(
        port=args.port,
        model=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        step=args.step,
        dataset=args.dataset,
        epoch=args.epoch,
        split_point=args.split_point,
        learning_rate=args.learning_rate,
        use_lora=args.lora,
        rate_mbps=args.mbps,
        micro_batch_size=args.micro_batch_size,
        offload_activation=args.offload_activation,
        offload_model_state=args.offload_model_state,
        offload_activation_mb_num=args.offload_activation_mb_num,
        offload_model_state_sp_num=args.offload_model_state_sp_num,
        sort_batch=args.sort,
        save_dir=args.save_dir,
        pipeline_mode=convert_pipeline_mode(args.pmode),
        max_client_mem_mb=args.max_client_mem_gb * 1024,
    )
    # 只要看到offload_activation_mb_num大于0，就默认开启offload_activation
    # 如果offload_activation, 则offload_activation_mb_num=batch_size/micro_batch_size
    if args.offload_activation:
        args.offload_activation_mb_num = (args.batch_size + args.micro_batch_size - 1) // args.micro_batch_size
    elif args.offload_activation_mb_num > 0:
        args.offload_activation = True
    if args.offload_model_state:
        args.offload_model_state_sp_num = args.split_point
    elif args.offload_model_state_sp_num > 0:
        args.offload_model_state_sp_num = max(0, min(args.split_point, args.offload_model_state_sp_num))
        args.offload_model_state = True

    run_client(args, profile)


if __name__ == "__main__":
    main()
