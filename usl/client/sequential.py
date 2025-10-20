import logging

import torch
from usl.client.client import Client

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import PreTrainedModel

from usl.client.client import Client, ClientArgs


class SequentialClientTrainer(Client):
    def __init__(
        self,
        client_args: ClientArgs,
        head_model: PreTrainedModel,
        tail_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        client_device: str,
        train_logger: logging.Logger,
        dataset_train: Dataset,
        dataset_test: Dataset,
        num_workers: int = 4,
        normalize_loss: bool = True,  # NEW: 按 accum_steps 归一化 loss
    ):
        super().__init__(
            client_args, head_model, tail_model, tokenizer, client_device, train_logger, dataset_train, dataset_test, num_workers, normalize_loss
        )

    def _train_minibatches(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx):
        batch_loss = 0

        # 1. Process each micro-batch sequentially
        for mb_idx in range(grad_accum_steps):
            #  Head forward and send
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)

            # 2. Wait for server activation
            server_activation_payload = self.activation_from_server_queue.get(block=True)
            activation_to_tail, loss = self._tail_fwd_micro(server_activation_payload)
            batch_loss += loss.item()

            # 3. Tail backward and send gradients
            grad_payload = self._tail_bwd_micro(
                loss,
                activation_to_tail,
                token=server_activation_payload.token,
                group_id=group_id,
                mb_idx=mb_idx,
                mb_total=grad_accum_steps,
            )
            self.gradient_to_server_queue.put(grad_payload)
            if mb_idx == grad_accum_steps - 1:
                self.optimizer_tail.step()
                self.optimizer_tail.zero_grad(set_to_none=True)
            # 4. Wait for server gradient
            server_grad_payload = self.gradient_from_server_queue.get(block=True)
            self._head_bwd_micro(server_grad_payload)
            if mb_idx == grad_accum_steps - 1:
                self.optimizer_head.step()
                self.optimizer_head.zero_grad(set_to_none=True)
        self.client_max_mem_alloc_mb = max(self.client_max_mem_alloc_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)

        return batch_loss
