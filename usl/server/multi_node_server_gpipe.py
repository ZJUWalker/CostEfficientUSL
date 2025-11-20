from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from usl.server.multi_node_server import PipelineServer
from usl.server.single_server import PipelineMode, ServerArgs, SingleServer
from usl.socket import SocketCommunicator, Payload
from usl.utils.usl_gantt_plot import GanttChartData
from usl.offload import CpuOffloadHookWithOffloadHandler, AsyncDoubleBufferGroupOffloadHandler
from usl.server.base import *
import threading


class GpipeServer(PipelineServer):

    def __init__(
        self,
        server_args: ServerArgs,
        server_model: nn.Module,
        optimizer_clz: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(server_args, server_model, optimizer_clz, logger)
        self.block = True
        self.curr_fwd_mb_idx = AtomicInt(0)
        self.curr_bwd_mb_idx = AtomicInt(0)
        self.activation_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}  # store fwd input and output activation for bwd

    def _recv_loop(self):
        while not self.stop_event.is_set():
            # 这里是阻塞式的 recv，只要有消息就会处理
            try:
                if self.curr_fwd_mb_idx.get() < self.num_micro_batches:
                    self._recv_mb_fwd_tensors(dtype=torch.float32, block=self.block)
                    # for _ in range(2):
                    #     # recv two parts : activation, attention_mask
                    #     self._recv_tensor(self.recv_fwd_acti_rank, dtype=torch.float32, block=self.block)
                else:
                    # recv only gradient
                    self._recv_mb_bwd_grads(dtype=torch.float32, block=self.block)
            except Empty:
                pass
            # except Exception as e:
            #     print(f"Error in _recv_loop: {e}")
            #     break
            time.sleep(0.001)

    def _send_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.curr_fwd_mb_idx.get() < self.num_micro_batches:
                    # # 发送前向传播的输入激活
                    # self._send_tensor(self.send_fwd_acti_rank, self.activation_from_pre_rank_queue, block=self.block)
                    # # 发送前向传播的attention mask
                    # self._send_tensor(self.send_fwd_acti_rank, self.attention_mask_queue, block=self.block)
                    # # 发送反向传播的梯度
                    self._send_mb_fwd_tensors(block=self.block)
                else:
                    self._send_mb_bwd_grads(block=self.block)
                    # self._send_tensor(self.send_bwd_grad_rank, self.gradient_to_pre_rank_queue, block=self.block)
            except Empty:
                pass
            # except Exception as e:
            #     print(f"Error in _send_loop: {e}")
            #     break
            time.sleep(0.001)
        pass

    def _compute_task(self):
        timeout = 30  # seconds,
        while not self.stop_event.is_set():
            if self.curr_fwd_mb_idx.get() < self.num_micro_batches:
                # do forward pass
                try:
                    # 1. get activation from queue
                    # print(f'[rank {self.rank}] try to get activation from pre rank')
                    msg_type, mb_idx, activation_from_pre_rank = self.activation_from_pre_rank_queue.get(block=self.block, timeout=timeout)
                    print(
                        f'Rank {self.rank} get activation from pre rank, mb_idx: {mb_idx}, activation_from_pre_rank: {activation_from_pre_rank.shape}'
                    )
                    assert msg_type == MSG_TYPE_ACTIVATION, 'msg_type is not activation!'
                    # 记录前向传播的输入和输出激活，用于后续的反向传播
                    msg_type, mb_idx_mask, attention_mask = self.attention_mask_from_pre_rank_queue.get(True, timeout=timeout)
                    print(f'Rank {self.rank} get attention mask, mb_idx: {mb_idx_mask}, attention_mask: {attention_mask.shape}')
                    assert msg_type == MSG_TYPE_ATTENTION_MASK, 'msg_type is not attention_mask!'
                    assert mb_idx == mb_idx_mask, 'acti and mask should have the same mb_idx!'
                    # 2. forward pass
                    activation_from_pre_rank = activation_from_pre_rank.requires_grad_(True)
                    activation_to_next_rank: torch.Tensor = self.trunk_model(
                        hidden_states=activation_from_pre_rank,
                        attention_mask=attention_mask,
                        position_embeddings=None,
                    )
                    # 3. put activation,attention_mask to queue
                    activation_to_next_rank = activation_to_next_rank.requires_grad_(True)

                    self.activation_to_next_rank_queue.put((MSG_TYPE_ACTIVATION, mb_idx, activation_to_next_rank))
                    self.attention_mask_to_next_rank_queue.put((MSG_TYPE_ATTENTION_MASK, mb_idx, attention_mask))
                    # 4. save input and output activation for bwd
                    self.activation_dict[mb_idx] = (activation_from_pre_rank, activation_to_next_rank)
                    # 5. update curr_fwd_mb_idx
                    self.curr_fwd_mb_idx.increment()
                    print(f"Rank {self.rank} finish forward pass for mb_idx {mb_idx}")
                except Empty:
                    # print(f"Rank {self.rank} Timeout when getting activation from pre rank,maybe the client is disconnected.")
                    pass
                except Exception as e:
                    print(f"Rank {self.rank} Error in _compute_task fwd phase: {e}")
                    break
                time.sleep(0.001)
            else:
                # do backward pass
                try:
                    # 1. get gradient from queue
                    msg_type, mb_idx, gradient_from_next_rank = self.gradient_from_next_rank_queue.get(block=self.block, timeout=timeout)
                    assert msg_type == MSG_TYPE_GRADIENT, 'msg_type is not gradient!'
                    try:
                        # 2. get input activation and output activation for bwd
                        fwd_input_acti, fwd_output_acti = self.activation_dict.pop(mb_idx)
                        # 3. do mb_idx backward
                        torch.autograd.backward(fwd_output_acti, gradient_from_next_rank, retain_graph=False)
                        # 4. put gradient to queue
                        self.gradient_to_pre_rank_queue.put((MSG_TYPE_GRADIENT, mb_idx, fwd_input_acti.grad))
                        # 5. update curr_bwd_mb_idx
                        self.curr_bwd_mb_idx.increment()
                        print(f"Rank {self.rank} finish backward pass for mb_idx {mb_idx}")
                        # 6. update global_batch_idx and update stage if all microbatches are done
                        if self.curr_fwd_mb_idx.get() == self.num_micro_batches and self.curr_bwd_mb_idx.get() == self.num_micro_batches:
                            self.curr_fwd_mb_idx.set(0)
                            self.curr_bwd_mb_idx.set(0)
                            self.global_batch_idx += 1
                            self._update_stage()
                            print(f"Rank {self.rank} update global_batch_idx to {self.global_batch_idx}")
                    except KeyError:
                        print(f"Rank {self.rank} Error in _compute_task bwd phase: activation not found for mb_idx {mb_idx}")
                        continue
                except Empty:
                    # print(f"Rank {self.rank} Timeout when getting gradient from next rank,maybe the client is disconnected.")
                    pass
                except Exception as e:
                    print(f"Error in _compute_task bwd phase: {e}")
                    break
            time.sleep(0.001)
        pass
