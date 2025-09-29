import os
import argparse
import warnings

import torch
from usl.client import Client, ClientArgs
from usl.utils.log_utils import create_logger
from usl.utils.load_utils import load_client, load_dataset
from usl.utils.exp import set_seed
from usl.server.single_server import PipelineMode, convert_pipeline_mode
from torch.distributed.pipelining import PipelineStage

# from deepspeed.ops.op_builder import AsyncIOBuilder


# nvme_handle = AsyncIOBuilder().load().aio_handle(block_size=2 * 1024 * 1024)

SEED = 0
warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)


def run_client(args: ClientArgs):
    set_seed(SEED)
    dataset_name = args.dataset
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    model_name = args.model
    model_dir = os.path.join("data/models", model_name)
    split_point = args.split_point
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log_dir = f"log/{args.model}/client"
    logger = create_logger(log_file_name="client.log", console_output=False, log_dir=log_dir)
    logger.info(f"client start with args: {args}")

    # ---------------load model and tokenizer --------------------------
    head, tail, tokenizer = load_client(model_dir, model_name, split_point, use_lora=False, use_qlora_4bit=False, use_qlora_8bit=False)
    # ---------------load dataset------------------------------------
    client_dataloaders = load_dataset(dataset_name, tokenizer, [0], batch_size, max_seq_len)
    dataloader = client_dataloaders[0]  # 默认只取第一个客户端数据

    # -----------------create client----------
    client = Client(
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
    client.train_epoch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8888, help="port to listen")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
    parser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-SL", "--max_seq_len", type=int, default=512, help="max sequence length")
    parser.add_argument("-S", "--step", type=int, default=5)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-E", "--epoch", type=int, default=1)
    parser.add_argument("-SP", "--split_point", type=int, default=4)
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mbps", type=int, default=0)
    parser.add_argument("--async_io", action="store_true", default=True)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--offload_activation", action="store_true", default=False)
    parser.add_argument("--offload_model_state", action="store_true", default=False)
    parser.add_argument("--sort_batch", action="store_true", default=False)
    parser.add_argument("--pmode", type=str, default="strict", help='mode of pipeline, "strict" or "loose"')

    args = parser.parse_args()
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
        rate_mbps=args.mbps,
        async_io=args.async_io,
        micro_batch_size=args.micro_batch_size,
        offload_activation=args.offload_activation,
        offload_model_state=args.offload_model_state,
        sort_batch=args.sort_batch,
        pipeline_mode=convert_pipeline_mode(args.pmode),
    )

    run_client(args)


if __name__ == "__main__":
    main()
