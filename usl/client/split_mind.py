import logging
from queue import Queue, Empty
import time
from typing import Dict, Optional
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import PreTrainedModel

from usl.client.client import Client, ClientArgs
from usl.socket import SocketCommunicator, Payload
from usl.utils.thread_safe_utils import AtomicInt, AtomicBool


class SplitMindClientTrainer(Client):

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
        self.is_head_fwd_done = AtomicBool(False)
        self.head_fwd_send_count = AtomicInt(0)

    @torch.no_grad()
    def _handle_client_rank_0_send(self):
        self._check_comm(self.communicator_rank_0)
        while not self.stop_event.is_set():
            try:
                if self.is_head_fwd_done.get() and self.activation_to_server_queue.empty():
                    time.sleep(0.001)
                    continue
                payload: Optional[Payload | Dict] = self.activation_to_server_queue.get(timeout=0.001)
                if payload is not None:  # 可能是 None（队列空）
                    start_send = time.time()
                    self.communicator_rank_0.send(payload)
                    end_time = time.time()
                    self.head_fwd_send_count.increment()
                    if isinstance(payload, dict) and "stop" in payload:
                        print("send stop flag")
                        continue
                    else:
                        # print(f'rank 0 ,communicator:{self.communicator_rank_0.conn},send payload: {payload.mb_idx}, {payload.is_activation}')
                        mb_idx = payload.mb_idx
                        self.profile_data[mb_idx].head_fwd_send_timestamp[0] = start_send
                        self.profile_data[mb_idx].head_fwd_send_timestamp[1] = end_time
                        if self.curr_step_idx > 0:
                            self.head_fwd_send_time += end_time - start_send
                else:
                    continue
            except Empty:
                pass
            time.sleep(0.001)  # 避免频繁发送
        print("client rank-0 send thread exit")
        pass

    @torch.no_grad()
    def _handle_client_rank_n_send(self):
        self._check_comm(self.communicator_rank_n)
        while not self.stop_event.is_set():
            if self.head_fwd_send_count.get() < self.num_minibatch:
                time.sleep(0.001)
                continue
            try:
                payload = self.gradient_to_server_queue.get(timeout=0.001)
                # print(f'rank n send payload: {payload.mb_idx}, {payload.is_activation}')
                if payload is not None:  # 可能是 None（队列空）
                    # print(f'send gradient payload')
                    self.sent_payload_bytes += payload.payload_nbytes()
                    start_send = time.time()
                    self.communicator_rank_n.send(payload)
                    end_time = time.time()
                    mb_idx = payload.mb_idx
                    self.profile_data[mb_idx].tail_bwd_send_timestamp[0] = start_send
                    self.profile_data[mb_idx].tail_bwd_send_timestamp[1] = end_time
                    if self.curr_step_idx > 0:
                        self.tail_bwd_send_time += end_time - start_send
                else:
                    continue
            except Empty:
                pass
            time.sleep(0.001)  # 避免频繁发送
        print("client rank-0 send thread exit")
        pass

    def _train_minibatches(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx):
        # 1. Head forward and send
        if self.offload_activation:
            self.activation_offload_handler.start_fwd()
        # self._check_mem_usage('before head fwd')
        for mb_idx in range(grad_accum_steps):
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)
        self.is_head_fwd_done.set(True)
        # self._check_mem_usage('after head fwd')
        # do offload and reload
        if self.offload_model_state:
            # print('offload and reload')
            # reload tail model and optimizer
            self.tail_model_manager.reload(True)
            self.tail_os_manager.reload(True)
            # offload head model and optimizer
            self.head_model_manager.offload(True)
            self.head_os_manager.offload(True)
            # wait for offload ,releasing GPU memory
            self.head_model_offload_timestamp = self.head_model_manager.wait_offload()
            # print('head model offload finished')
            self.head_optimizer_offload_timestamp = self.head_os_manager.wait_offload()
            # wait for reload,
            self.tail_model_reload_timestamp = self.tail_model_manager.wait_reload()
            # print('tail model reload finished')
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
        # self._check_mem_usage('after all tail fwd&bwd')
        # 3. Tail model step
        if self.offload_model_state:
            # wait for tail optimizer reload,or else it will cause error when step
            # self.tail_os_mgr.wait_reload()
            self.tail_optimizer_reload_timestamp = self.tail_os_manager.wait_reload()
        self.optimizer_tail.step()
        self.optimizer_tail.zero_grad(set_to_none=True)
        # self._check_mem_usage('after tail step')
        if self.offload_model_state:
            self.head_model_manager.reload(True)
            self.head_os_manager.reload(True)
            self.tail_model_manager.offload(True)
            self.tail_os_manager.offload(True)
            self.head_model_reload_timestamp = self.head_model_manager.wait_reload()
            # print('head model reload finished')
            self.tail_model_offload_timestamp = self.tail_model_manager.wait_offload()
            # print('tail model offload finished')
            self.tail_optimizer_offload_timestamp = self.tail_os_manager.wait_offload()
        # self._check_mem_usage('after tail step and offload/reload')
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
            # self.head_os_mgr.wait_reload()
            self.head_optimizer_reload_timestamp = self.head_os_manager.wait_reload()
        self.optimizer_head.step()
        self.optimizer_head.zero_grad(set_to_none=True)

        # 6. Reset status
        self.is_head_fwd_done.set(False)
        self.head_fwd_send_count.set(0)
        if self.offload_model_state:
            self.head_model_manager.update_param_ptr()
        # 7. Memory tracking
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
