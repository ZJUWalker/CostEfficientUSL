import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from usl.socket import SocketCommunicator
from usl.utils.dataset.exp import AverageMeter
from typing import Dict, List, Optional, Tuple
from usl.utils.tensor_utils import pad_inputs
from dataclasses import dataclass
import argparse


@dataclass
class ClientArgs:
    port: int = 8000
    model: str = "meta-llama/llama3.2-1b"
    batch_size: int = 4
    max_seq_len: int = 256
    step: int = 10
    dataset: str = "gsm8k"
    epoch: int = 1
    split_point: int = 2
    learning_rate: float = 5e-4
    rate_mbps: float = 10  # rate in Mbps
    micro_batch_size: int = 1
    async_io: bool = False


class Client(object):

    def __init__(
        self,
        client_args: ClientArgs,  # ✅ 改成 ClientArgs 类型
        head_model: nn.Module,
        tail_model: nn.Module,
        tokenizer: AutoTokenizer,
        client_device: str,
        train_logger: logging.Logger,
        dataset_train: Dataset,
        dataset_test: Dataset,
    ):
        self.client_device = client_device
        self.client_args = client_args
        self.head_model = head_model.to(self.client_device)
        self.tail_model = tail_model.to(self.client_device)
        self.tokenizer = tokenizer
        self.train_logger = train_logger
        self.train_loader = dataset_train
        self.test_loader = dataset_test
        self.local_ep = client_args.epoch  # ✅ 点操作符访问
        self.lr = client_args.learning_rate  # ✅ 点操作符访问

        print(
            f"[Client] after model loaded, cuda memory: {torch.cuda.memory_allocated(device=client_device) / 1024**3:.4f} GB, "
            f"max memory: {torch.cuda.max_memory_allocated(device=client_device) / 1024**3:.4f} GB"
        )

        self.optimizer_head = torch.optim.Adam(self.head_model.parameters(), lr=self.lr)
        self.optimizer_tail = torch.optim.Adam(self.tail_model.parameters(), lr=self.lr)
        self.avg_loss = AverageMeter()
        self.compute_time = 0

    def _forward(
        self, client_conn: SocketCommunicator, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[CausalLMOutputWithPast, torch.Tensor, torch.Tensor]:
        # forward head
        head_outs: Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]] = self.head_model.forward(input_ids, attention_mask)
        head_outs[0].requires_grad_(True)
        attention_mask = head_outs[1].cpu() if head_outs[1] is not None else None
        head_out_to_server = {
            "activation": head_outs[0].cpu(),
            "attention_mask": attention_mask,
            "position_embeddings": ([ho.float().cpu() for ho in head_outs[2]] if len(head_outs) > 2 else None),
            "is_training": True,
        }
        if head_out_to_server["attention_mask"] is None:
            head_out_to_server.pop("attention_mask")
        if head_out_to_server["position_embeddings"] is None:
            head_out_to_server.pop("position_embeddings")
        client_conn.send(head_out_to_server)
        server_forward_output = client_conn.receive()

        activation_to_tail = torch.tensor(
            server_forward_output["server_activation"],
            device=self.client_device,
            dtype=head_outs[0].dtype,
            requires_grad=True,
        )
        output = self.tail_model.forward(
            hidden_states=activation_to_tail,
            attention_mask=head_outs[1] if head_outs[1] is not None else None,
            position_embeddings=head_outs[2] if len(head_outs) > 2 else None,
            labels=labels,
        )
        return output, activation_to_tail, head_outs[0]

    def _backward(self, client_conn: SocketCommunicator, loss: torch.Tensor, activation_to_tail: torch.Tensor, head_output_activation: torch.Tensor):
        # tail model backward
        loss.backward()
        # get grads of tail model input activation
        grads_to_server = activation_to_tail.grad.cpu()
        # send grads to server for server backward
        tail_grads_to_server = {"gradient": grads_to_server}
        client_conn.send(tail_grads_to_server)
        # receive server backward output grad
        server_backward_output = client_conn.receive()
        grads_from_server = torch.tensor(
            server_backward_output["server_gradient"],
            device=self.client_device,
            dtype=activation_to_tail.dtype,
        )
        # head model backward
        head_output_activation.backward(grads_from_server)
        # optimizer step
        self.optimizer_head.step()
        self.optimizer_tail.step()
        self.optimizer_head.zero_grad()
        self.optimizer_tail.zero_grad()
        pass

    def train_epoch(self):
        """
        Train the client model
        """
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        self.avg_loss.reset()
        # ❗注意：ClientArgs 里没有 batch_per_sync，需要加一个默认值或删掉
        with SocketCommunicator(
            host="localhost",
            port=self.client_args.port,
            is_server=False,
            rate_limit_mbps=self.client_args.rate_mbps,
        ) as client_conn:
            start_time = time.time()
            for epoch in range(self.local_ep):
                self.train_logger.info(f"[Client] start train epoch {epoch+1}, data loader len: {len(self.train_loader)}")
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    loss = self.train_batch(batch, client_conn, batch_idx)
                    self.train_logger.info(f"[Client] train epoch {epoch+1}, batch {batch_idx}/{len(self.train_loader)}, loss: {loss:.4f}")
                    if batch_idx == 20:
                        break
            end_time = time.time()
            self.train_logger.info(f"[Client Finished] train epoch time: {end_time - start_time:.2f} s, compute time: {self.compute_time:.2f} s")

    def train_batch(self, batch: Dict, client_conn: SocketCommunicator, batch_idx: int):
        input_ids = batch["input_ids"].to(self.client_device)
        attention_mask = batch["attention_mask"].to(self.client_device)
        input_ids, attention_mask = pad_inputs(input_ids, attention_mask, self.client_args.max_seq_len)
        labels = input_ids
        # forward
        output, activation_to_tail, head_output_activation = self._forward(client_conn, input_ids, attention_mask, labels)
        # backward
        loss = output.loss
        self.avg_loss.update(loss.item())
        self._backward(
            client_conn,
            loss,
            activation_to_tail,
            head_output_activation,
        )
        torch.cuda.empty_cache()
        return loss
