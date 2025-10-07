import logging
from queue import Queue, Empty
from typing import Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import PreTrainedModel

from usl.client.client import Client, ClientArgs


class GPipeClientTrainer(Client):
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
        normalize_loss: bool = True,
    ):
        super().__init__(
            client_args, head_model, tail_model, tokenizer, client_device, train_logger, dataset_train, dataset_test, num_workers, normalize_loss
        )

    def _train_minibatches(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx):
        if self.offload_activation:
            self.activation_offload_handler.start_fwd()
        for mb_idx in range(grad_accum_steps):
            #  head fwd and send
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)
            pass
        # do offload and reload
        if self.offload_model_state:
            # reload tail model and optimizer
            self.tail_m_mgr.reload(True)
            self.tail_os_mgr.reload(True)
            # offload head model and optimizer
            self.head_m_mgr.offload(True)
            self.head_os_mgr.offload(True)
            # wait for offload ,releasing GPU memory
            self.head_m_mgr.wait_offload()
            self.head_os_mgr.wait_offload()
            # wait for reload,
            self.tail_m_mgr.wait_reload()
        batch_loss = 0
        # tail fwd
        no_tail_fwd_mb_list = [False] * grad_accum_steps
        activation_to_tail_queue: Queue[Dict] = Queue(maxsize=grad_accum_steps)
        while True:
            if not all(no_tail_fwd_mb_list) and not self.stop_event.is_set():
                try:
                    server_activation_payload = self.activation_from_server_queue.get(timeout=0.001)
                    if server_activation_payload is not None:
                        mb_idx = server_activation_payload.mb_idx
                        no_tail_fwd_mb_list[mb_idx] = True
                        activation_to_tail, loss = self._tail_fwd_micro(server_activation_payload)
                        batch_loss += loss.item()
                        activation_to_tail_queue.put(
                            {
                                "mb_idx": mb_idx,
                                "activation": activation_to_tail,
                                "loss": loss,
                                "token": server_activation_payload.token,
                            }
                        )
                except Empty:
                    continue
            else:
                break
            pass
        # tail bwd
        no_tail_bwd_mb_list = [False] * grad_accum_steps
        while True:
            if not all(no_tail_bwd_mb_list) and not self.stop_event.is_set():
                try:
                    activation_to_tail_dict = activation_to_tail_queue.get(block=True)
                    if activation_to_tail_dict is not None:
                        mb_idx = activation_to_tail_dict["mb_idx"]
                        loss = activation_to_tail_dict["loss"]
                        no_tail_bwd_mb_list[mb_idx] = True
                        try:
                            grad_payload = self._tail_bwd_micro(
                                loss,
                                activation_to_tail_dict["activation"],
                                activation_to_tail_dict["token"],
                                group_id,
                                mb_idx,
                                grad_accum_steps,
                            )
                        except Exception as e:
                            print(f"error when send grad payload: {e}")
                        self.gradient_to_server_queue.put(grad_payload)
                except Empty:
                    continue
            else:
                break
            pass
        # tail step
        if self.offload_model_state:
            # wait for tail optimizer reload,or else it will cause error when step
            self.tail_os_mgr.wait_reload()
        self.optimizer_tail.step()
        self.optimizer_tail.zero_grad(set_to_none=True)
        # head bwd
        # offload tail model and optimizer
        if self.offload_model_state:
            self.head_m_mgr.reload(True)
            self.head_os_mgr.reload(True)
            self.tail_m_mgr.offload(True)
            self.tail_os_mgr.offload(True)
            self.head_m_mgr.wait_reload()
            self.tail_m_mgr.wait_offload()
            self.tail_os_mgr.wait_offload()
        if self.offload_activation:
            self.activation_offload_handler.start_bwd()
        no_head_bwd_mb_list = [False] * grad_accum_steps
        while True:
            if not all(no_head_bwd_mb_list) and not self.stop_event.is_set():
                try:
                    server_grad_payload = self.gradient_from_server_queue.get(timeout=0.001)
                    if server_grad_payload is not None:
                        mb_idx = server_grad_payload.mb_idx
                        no_head_bwd_mb_list[mb_idx] = True
                        self._head_bwd_micro(server_grad_payload)
                except Empty:
                    continue
            else:
                break
            pass
        if self.offload_model_state:
            self.head_os_mgr.wait_reload()
        self.optimizer_head.step()
        self.optimizer_head.zero_grad(set_to_none=True)
        self.max_mem_allocated_mb = max(self.max_mem_allocated_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)
        return batch_loss
