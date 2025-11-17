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

from usl.server.single_server import PipelineMode, ServerArgs, SingleServer
from usl.socket import SocketCommunicator, Payload
from usl.utils.usl_gantt_plot import GanttChartData
from usl.offload import CpuOffloadHookWithOffloadHandler, AsyncDoubleBufferGroupOffloadHandler


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading

"""
Implementation of a thread-safe counter with many producers and many consumers.
"""


class Counter:
    def __init__(self, initial_count):
        self.count = initial_count
        self.cv = threading.Condition()

    def decrement(self):
        self.cv.acquire()
        self.count -= 1
        self.cv.notify_all()
        self.cv.release()

    def wait(self):
        self.cv.acquire()
        while self.count > 0:
            self.cv.wait()
        self.cv.release()


class PipelineServer:

    def __init__(
        self,
        server_args: ServerArgs,
        server_model: nn.Module,
        optimizer_clz: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        logger: Optional[logging.Logger] = None,
    ):

        # ---- Config & logging
        self.server_args = server_args
        self.server_device = torch.cuda.device(dist.get_rank())
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # ---- Profile Metrics
        self.max_cuda_memory_allocated = 0.0
        self.profile_data: List[GanttChartData] = []
        self.global_step = 0
        self.server_fwd_time = 0
        self.server_fwd_send_time = 0
        self.server_bwd_time = 0
        self.server_bwd_send_time = 0
        self.activation_offload_time = 0

        # ---- Executors & coordination
        self.main_executor: Optional[ThreadPoolExecutor] = None
        self.compute_future: Optional[Future] = None
        self.client_future: Optional[Future] = None
        self.server_future: Optional[Future] = None
        self.stop_event = threading.Event()

        # ---- CUDA streams
        torch.cuda.set_stream(torch.cuda.Stream(self.server_device))  # set cuda compute stream
        self.load_stream = torch.cuda.Stream(self.server_device)  # set cuda load stream
        self.offload_stream = torch.cuda.Stream(self.server_device)  # set cuda offload stream
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        # ---- Model & optimizer
        self.lr = server_args.learning_rate
        self.trunk_model = server_model.to(self.server_device).train()
        self.optimizer: torch.optim.Optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)
        self.offload_activation_mb_num = self.server_args.offload_activation_mb_num
        if self.offload_activation_mb_num > 0:
            self.activation_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
                num_minibatch=self.offload_activation_mb_num,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
            )
            self.activation_offload_ctx = CpuOffloadHookWithOffloadHandler(self.activation_offload_handler)
        # init communication and pipeline info
        self._init_comminication()
        self._init_pipeline_info()

    def _init_comminication(self):
        # ---- Communicator
        # the first and last node need a socket to communicate with the client
        if self.rank == 0 or self.rank == self.world_size - 1:
            self.communicator = SocketCommunicator(
                is_server=True,
                port=self.server_args.port if self.rank == 0 else self.server_args.port + 1,  # different port for each node
                buffer_size=1024 * 4,  # 4KB
                rate_limit_mbps=self.server_args.rate_limit_mbps,
            )
        # mutil processing node server(GPU2GPU communication)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        dist.init_process_group(backend="nccl")

        # ---- Queues (compute pipeline)
        self.activation_from_pre_rank_queue: Queue = Queue()
        self.activation_to_next_rank_queue: Queue = Queue()
        self.gradient_from_next_rank_queue: Queue = Queue()
        self.gradient_to_pre_rank_queue: Queue = Queue()

        # ---- Micro-batch context/state
        # self.ctx: Dict[str, Dict[str, torch.Tensor]] = {}
        # self.group_state: Dict[str, Dict[str, Any]] = {}

    def _init_pipeline_info(self):
        # ---- Pipeline info
        self.pipeline_mode = self.server_args.pipeline_mode
        self.batch_size = self.server_args.batch_size
        self.micro_batch_size = self.server_args.micro_batch_size
        self.num_micro_batches = (self.batch_size + self.micro_batch_size - 1) // self.micro_batch_size
        self.curr_mb_idx = 0

    def _is_last_mb_of_global_batch(self) -> bool:
        return self.curr_mb_idx == self.num_micro_batches - 1

    def _init_param_efficient_mode(self):
        pass

    # --------------- Public lifecycle ---------------
    def run(self):
        """Start compute loop and accept a single client, all via executor workers."""
        self.main_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="server")
        # Accept a single client (blocking here; compute loop runs on executor)
        if self.rank == 0 or self.rank == self.world_size - 1:
            self.communicator.accept_client()
            self.logger.info(f"Rank {self.rank} Connected from {self.communicator.conn}")
        # Start client communication loop on executor
        self.client_future = self.main_executor.submit(self._recv_loop)
        self.server_future = self.main_executor.submit(self._send_loop)
        # Start compute loop immediately
        self.compute_future = self.main_executor.submit(self._compute_loop)
        # break
        if self.rank == 0:
            self.logger.info("Client connected")

    def _compute_loop(self):
        try:
            if self.server_args.pipeline_mode in [PipelineMode.GPIPE, PipelineMode.PIPE_DREAM_WC]:
                self._compute_task_gpipe_or_pipe_dream_wc()
            elif self.server_args.pipeline_mode == PipelineMode.PIPE_DREAM_STRICT:
                self._compute_task_pipedream_strict()
            elif self.server_args.pipeline_mode == PipelineMode.NAIVE:
                self._compute_task_sequential()
            else:
                raise ValueError(f"Unknown pipeline mode: {self.server_args.pipeline_mode}")
        except Exception as e:
            self.logger.exception(f"Compute loop crashed: {e}")
            self.shutdown()

    def _recv_loop(self):
        pass

    def _send_loop(self):
        pass

    def shutdown(self):
        """Graceful shutdown for futures, sockets, and executor."""
        if not self.stop_event.is_set():
            print(f"Rank {self.rank} Shutting down server...")
            # 1. set stop event to signal loops to exit
            self.stop_event.set()
            # 2. wait for loops to exit
            self.client_future.result()
            self.server_future.result()
            self.compute_future.result()
            # 3. close sockets if open
            if self.rank == 0 or self.rank == self.world_size - 1:
                self.communicator.close()
            # 4. shutdown thread pool executor
            if self.main_executor:
                self.main_executor.shutdown(wait=True, cancel_futures=True)
        # 5.close distributed process group
        dist.destroy_process_group()

    @property
    def recv_fwd_acti_rank(self) -> int:
        return self.rank - 1

    @property
    def recv_bwd_grad_rank(self) -> int:
        return self.rank + 1

    @property
    def send_fwd_acti_rank(self) -> int:
        return self.rank + 1

    @property
    def send_bwd_grad_rank(self) -> int:
        return self.rank - 1  # same as recv_fwd_acti_rank

    def _send_meta(self, payload: Optional[Payload | torch.Tensor], rank: int):

        pass

    def _recv_meta(self, rank: int) -> Optional[Payload | torch.Tensor]:

        pass

    def _send_tensor(self, tensor: torch.Tensor, dst: int):
        # --- 1. 发送 shape ---
        shape_tensor = torch.tensor(list(tensor.shape), dtype=torch.int32, device=tensor.device)

        # 设定全局的张量dim为3 [batch_size, seq_len, hidden_size],包括attention_mask,position_embeddings
        # 告诉对方每一维的大小
        dist.send(shape_tensor, dst)

        # --- 2. 发送数据 ---
        flat_tensor = tensor.contiguous().view(-1)
        dist.send(flat_tensor, dst)

    def _compute_task_gpipe_or_pipe_dream_wc(self):

        pass

    def _compute_task_pipedream_strict(self):

        pass

    def _compute_task_sequential(self):

        pass
