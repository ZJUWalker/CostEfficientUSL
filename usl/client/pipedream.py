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
from usl.utils.thread_safe_utils import AtomicBool, AtomicInt


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
        self.server_stage_num = client_args.server_world_size
        self.warmup_steps = None
        # self.is_sending_activation = AtomicBool(False)

        #     self.send_gradient = AtomicBool(False)
        # self.curr_send_activation_mb_count = AtomicInt(0)

    #     self.curr_recv_activation_mb_count = AtomicInt(0)
    #     self.curr_send_gradient_mb_count = AtomicInt(0)
    #     self.curr_recv_gradient_mb_count = AtomicInt(0)

    # def _reset(self):
    #     self.send_activation.set(True)
    #     self.send_gradient.set(False)
    #     self.curr_send_activation_mb_count.set(0)
    #     self.curr_recv_activation_mb_count.set(0)
    #     self.curr_send_gradient_mb_count.set(0)
    #     self.curr_recv_gradient_mb_count.set(0)

    @torch.no_grad()
    def _handle_client_rank_0_send(self):
        self._check_comm(self.communicator_rank_0)
        while not self.stop_event.is_set():
            try:
                payload: Optional[Payload | Dict] = self.activation_to_server_queue.get(timeout=0.001)
                if payload is not None:  # 可能是 None（队列空）
                    # self.is_sending_activation.set(True)
                    start_send = time.time()
                    self.communicator_rank_0.send(payload)
                    end_time = time.time()
                    # self.is_sending_activation.set(False)
                    # self.curr_send_activation_mb_count.increment()
                    # print(f"client rank 0 send activation, mb_idx: {payload.mb_idx}, mb_total: {payload.mb_total}")
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
            try:
                # if self.is_sending_activation.get() or self.curr_send_activation_mb_count.get() < self.warmup_steps:
                #     time.sleep(0.001)
                #     continue
                payload = self.gradient_to_server_queue.get(timeout=0.001)
                # print(f'rank n send payload: {payload.mb_idx}, {payload.is_activation}')
                if payload is not None:  # 可能是 None（队列空）
                    # print(f'send gradient payload')
                    self.sent_payload_bytes += payload.payload_nbytes()
                    start_send = time.time()
                    self.communicator_rank_n.send(payload)
                    end_time = time.time()
                    # print(f"client rank n send gradient, mb_idx: {payload.mb_idx}, mb_total: {payload.mb_total}")
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

    # TODO: 处理recv操作
    @torch.no_grad()
    def _handle_server_rank_0_send(self):
        self._check_comm(self.communicator_rank_0)
        while not self.stop_event.is_set():
            try:
                data: Optional[Dict | Payload] = self.communicator_rank_0.receive()
            except Exception as e:
                break
            if data is None:
                break
            if isinstance(data, dict) and "profile" in data:
                print(f"get profile data")
                try:
                    if self.client_max_mem_alloc_mb is not None and self.client_max_mem_alloc_mb > self.client_args.max_client_mem_mb:
                        print(f"client max mem alloc {self.client_max_mem_alloc_mb} > {self.client_args.max_client_mem_mb}, exit")
                    else:
                        # print(f'get profile data: {data},stop training')
                        self._save_profile_res(data)
                except Exception as e:
                    print(f"error when save profile data: {e}")
                finally:
                    self.stop_event.set()
                    break
            # print(f"client rank 0 recv gradient, mb_idx: {data.mb_idx}, mb_total: {data.mb_total}")
            # print(f'rank 0 recv payload: {data.mb_idx}, {data.is_activation}')
            assert not data.is_activation, "rank n recv data should be gradient"
            data.tensor = data.tensor.pin_memory()
            self.gradient_from_server_queue.put(data)
            time.sleep(0.001)  # 避免频繁发送

        print("server rank 0 send thread exit")

    @torch.no_grad()
    def _handle_server_rank_n_send(self):
        self._check_comm(self.communicator_rank_n)
        while not self.stop_event.is_set():
            try:
                data: Optional[Dict | Payload] = self.communicator_rank_n.receive()
                # print(f"client rank n recv activation, mb_idx: {data.mb_idx}, mb_total: {data.mb_total}")
            except Exception as e:
                print(f"server rank n recv error: {e}")
                break
            if data is None:
                print(f"server rank n recv None")
                break
            # print(f'rank n recv payload: {data.mb_idx}, {data.is_activation}')
            assert data.is_activation, "rank 0 recv data should be activation"
            data.tensor = data.tensor.pin_memory()
            self.activation_from_server_queue.put(data)
            time.sleep(0.001)  # 避免频繁发送
        print("server rank n send thread exit")

    def _train_minibatches(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx):
        self.warmup_steps = min(grad_accum_steps, 2 + self.server_stage_num)
        curr_head_fwd_mb_idx = 0

        # 1. Warmup phase for head model forward
        for mb_idx in range(self.warmup_steps):
            # print(f'client do head forward for mb_idx {mb_idx}')
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
                    server_activation_payload = self.activation_from_server_queue.get()
                    if server_activation_payload is not None:
                        mb_idx = server_activation_payload.mb_idx
                        no_tail_fwd_bwd_mb_list[mb_idx] = True
                        # print(f'client do tail forward&backward for mb_idx {mb_idx}')
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
                    server_grad_payload = self.gradient_from_server_queue.get()
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
        # self._reset()
        # self.curr_send_activation_mb_count.set(0)

        # 4. Memory tracking
        self.client_max_mem_alloc_mb = max(self.client_max_mem_alloc_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)
        return batch_loss
