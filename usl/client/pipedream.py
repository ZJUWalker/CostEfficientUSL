import logging
from queue import Queue, Empty
import time
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import PreTrainedModel

from usl.client.client import Client, ClientArgs


class PipeDreamStrictClientTrainer(Client):

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
        warmup_steps = grad_accum_steps // 2
        curr_head_fwd_mb_idx = 0

        # 1. Warmup phase for head model forward
        for mb_idx in range(warmup_steps):
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)
            curr_head_fwd_mb_idx += 1

        batch_loss = 0
        no_tail_fwd_bwd_mb_list = [False] * grad_accum_steps
        no_head_bwd_mb_list = [False] * grad_accum_steps

        # 2. Strict 1F1B phase for tail model forward and backward
        while not all(no_tail_fwd_bwd_mb_list) and not all(no_head_bwd_mb_list) and not self.stop_event.is_set():
            # Tail operation
            if not all(no_tail_fwd_bwd_mb_list):
                try:
                    server_activation_payload = self.activation_from_server_queue.get(block=True)
                    if server_activation_payload is not None:
                        mb_idx = server_activation_payload.mb_idx
                        no_tail_fwd_bwd_mb_list[mb_idx] = True
                        activation_to_tail, loss = self._tail_fwd_micro(server_activation_payload)
                        batch_loss += loss.item()
                        grad_payload = self._tail_bwd_micro(
                            loss,
                            activation_to_tail,
                            token=server_activation_payload.token,
                            group_id=group_id,
                            mb_idx=mb_idx,
                            mb_total=grad_accum_steps,
                        )
                        self.gradient_to_server_queue.put(grad_payload)
                except Empty:
                    pass

            # Head operation
            if not all(no_head_bwd_mb_list):
                try:
                    server_grad_payload = self.gradient_from_server_queue.get(block=True)
                    if server_grad_payload is not None:
                        mb_idx = server_grad_payload.mb_idx
                        no_head_bwd_mb_list[mb_idx] = True
                        self._head_bwd_micro(server_grad_payload)

                        # Head forward for the next micro-batch if available
                        if curr_head_fwd_mb_idx < grad_accum_steps:
                            payload = self._head_fwd_micro(
                                group_id,
                                curr_head_fwd_mb_idx,
                                grad_accum_steps,
                                micro_inputs[curr_head_fwd_mb_idx],
                                micro_masks[curr_head_fwd_mb_idx],
                                micro_labels[curr_head_fwd_mb_idx],
                            )
                            self.activation_to_server_queue.put(payload)
                            self.labels_dict[curr_head_fwd_mb_idx] = micro_labels[curr_head_fwd_mb_idx]
                            curr_head_fwd_mb_idx += 1
                except Empty:
                    pass

            time.sleep(0.001)

        # 3. Model step for tail and head models
        self.optimizer_tail.step()
        self.optimizer_tail.zero_grad(set_to_none=True)
        self.optimizer_head.step()
        self.optimizer_head.zero_grad(set_to_none=True)

        # 4. Memory tracking
        self.client_max_mem_alloc_mb = max(self.client_max_mem_alloc_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)
        return batch_loss


class PipeDreamWCClientTrainer(Client):

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
        # 1. Head forward and send
        if self.offload_activation:
            self.activation_offload_handler.start_fwd()
        self._check_mem_usage('before head fwd')
        for mb_idx in range(grad_accum_steps):
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)
        self._check_mem_usage('after head fwd')
        # do offload and reload
        if self.offload_model_state:
            # reload tail model and optimizer
            self.tail_m_mgr.reload(True)
            self.tail_os_mgr.reload(True)
            # offload head model and optimizer
            self.head_m_mgr.offload(True)
            self.head_os_mgr.offload(True)
            # wait for offload ,releasing GPU memory
            self.head_model_offload_timestamp = self.head_m_mgr.wait_offload()
            self.head_os_mgr.wait_offload()
            # wait for reload,
            self.tail_model_reload_timestamp = self.tail_m_mgr.wait_reload()
        batch_loss = 0
        no_tail_fwd_bwd_mb_list = [False] * grad_accum_steps
        no_head_bwd_mb_list = [False] * grad_accum_steps

        # 2. Tail forward and backward
        while True:
            if not all(no_tail_fwd_bwd_mb_list) and not self.stop_event.is_set():
                try:
                    server_activation_payload = self.activation_from_server_queue.get(timeout=0.001)
                    if server_activation_payload is not None:
                        mb_idx = server_activation_payload.mb_idx
                        no_tail_fwd_bwd_mb_list[mb_idx] = True
                        activation_to_tail, loss = self._tail_fwd_micro(server_activation_payload)
                        batch_loss += loss.item()
                        grad_payload = self._tail_bwd_micro(
                            loss,
                            activation_to_tail,
                            token=server_activation_payload.token,
                            group_id=group_id,
                            mb_idx=mb_idx,
                            mb_total=grad_accum_steps,
                        )
                        self.gradient_to_server_queue.put(grad_payload)
                except Empty:
                    pass
            else:
                break
        self._check_mem_usage('after all tail fwd&bwd')
        # 3. Tail model step
        if self.offload_model_state:
            # wait for tail optimizer reload,or else it will cause error when step
            self.tail_os_mgr.wait_reload()
        self.optimizer_tail.step()
        self.optimizer_tail.zero_grad(set_to_none=True)
        if self.offload_model_state:
            self.head_m_mgr.reload(True)
            self.head_os_mgr.reload(True)
            self.tail_m_mgr.offload(True)
            self.tail_os_mgr.offload(True)
            self.head_model_reload_timestamp = self.head_m_mgr.wait_reload()
            self.tail_model_offload_timestamp = self.tail_m_mgr.wait_offload()
            self.tail_os_mgr.wait_offload()
        # 4. Head backward
        if self.offload_activation:
            self.activation_offload_handler.start_bwd()
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
            time.sleep(0.001)

        # 5. Head model step
        if self.offload_model_state:
            self.head_os_mgr.wait_reload()
        self.optimizer_head.step()
        self.optimizer_head.zero_grad(set_to_none=True)

        # 6. Memory tracking
        self.client_max_mem_alloc_mb = max(self.client_max_mem_alloc_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)
        return batch_loss


class PipeDreamWCEagerClientTrainer(Client):
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
        # 1. Head forward and send
        for mb_idx in range(grad_accum_steps):
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)

        batch_loss = 0
        no_tail_fwd_bwd_mb_list = [False] * grad_accum_steps
        no_head_bwd_mb_list = [False] * grad_accum_steps

        while True:
            if not all(no_tail_fwd_bwd_mb_list) and not self.stop_event.is_set():
                try:
                    server_activation_payload = self.activation_from_server_queue.get(timeout=0.001)
                    if server_activation_payload is not None:
                        mb_idx = server_activation_payload.mb_idx
                        no_tail_fwd_bwd_mb_list[mb_idx] = True
                        activation_to_tail, loss = self._tail_fwd_micro(server_activation_payload)
                        batch_loss += loss.item()
                        grad_payload = self._tail_bwd_micro(
                            loss,
                            activation_to_tail,
                            token=server_activation_payload.token,
                            group_id=group_id,
                            mb_idx=mb_idx,
                            mb_total=grad_accum_steps,
                        )
                        self.gradient_to_server_queue.put(grad_payload)
                except Empty:
                    pass

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
            time.sleep(0.001)

        # 2. Model step
        self.optimizer_tail.step()
        self.optimizer_tail.zero_grad(set_to_none=True)
        self.optimizer_head.step()
        self.optimizer_head.zero_grad(set_to_none=True)

        # 3. Memory tracking
        self.client_max_mem_alloc_mb = max(self.client_max_mem_alloc_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)
        return batch_loss
