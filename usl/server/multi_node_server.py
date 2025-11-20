from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from abc import abstractmethod, ABC
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

from usl.server.base import *
from usl.server.single_server import PipelineMode, ServerArgs, SingleServer
from usl.socket import SocketCommunicator, Payload
from usl.utils.usl_gantt_plot import GanttChartData
from usl.offload import CpuOffloadHookWithOffloadHandler, AsyncDoubleBufferGroupOffloadHandler

import threading


class PipelineServer(ABC):

    def __init__(
        self,
        server_args: ServerArgs,
        server_model: nn.Module,
        optimizer_clz: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        logger: Optional[logging.Logger] = None,
    ):

        # ---- Config & logging
        self.server_args = server_args
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # ---- Profile Metrics
        self.max_cuda_memory_allocated = 0.0
        self.profile_data: List[GanttChartData] = []
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
        # ---- init communication and pipeline info
        self._init_comminication()
        # ---- CUDA streams
        self.device = f'cuda:{self.rank}'
        print(f"Rank {self.rank} Device {self.device}")
        torch.cuda.set_stream(torch.cuda.Stream(self.device))  # set cuda compute stream
        self.load_stream = torch.cuda.Stream(self.device)  # set cuda load stream
        self.offload_stream = torch.cuda.Stream(self.device)  # set cuda offload stream
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        # ---- Model & optimizer
        self.lr = server_args.learning_rate
        self.trunk_model = server_model
        self.trunk_model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)
        self.offload_activation_mb_num = self.server_args.offload_activation_mb_num
        if self.offload_activation_mb_num > 0:
            self.activation_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
                num_minibatch=self.offload_activation_mb_num,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
            )
            self.activation_offload_ctx = CpuOffloadHookWithOffloadHandler(self.activation_offload_handler)

    def _init_comminication(self):
        # thread-safe queues for tensor send and recv
        self._init_pipeline_info()
        self._init_tensor_queues()
        # ---- Communicator
        # mutil processing node server(GPU2GPU communication)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # the first and last node need a socket to communicate with the client
        if self.rank == 0 or self.rank == self.world_size - 1:
            self.communicator = SocketCommunicator(
                is_server=True,
                port=self.server_args.port if self.rank == 0 else self.server_args.port + 1,  # different port for each node
                buffer_size=1024 * 4,  # 4KB
                rate_limit_mbps=self.server_args.rate_limit_mbps,
            )
        # ---- Micro-batch context/state
        # self.ctx: Dict[str, Dict[str, torch.Tensor]] = {}
        # self.group_state: Dict[str, Dict[str, Any]] = {}

    def _init_tensor_queues(self):
        # ---- Queues (compute pipeline)
        # FWD activation queues
        # print(self.num_micro_batches)
        self.activation_from_pre_rank_queue: Queue[Tuple[int, int, torch.Tensor]] = Queue(maxsize=self.num_micro_batches)
        self.activation_to_next_rank_queue: Queue[Tuple[int, int, torch.Tensor]] = Queue(maxsize=self.num_micro_batches)
        # BWD gradient queues
        self.gradient_from_next_rank_queue: Queue[Tuple[int, int, torch.Tensor]] = Queue(maxsize=self.num_micro_batches)
        self.gradient_to_pre_rank_queue: Queue[Tuple[int, int, torch.Tensor]] = Queue(maxsize=self.num_micro_batches)
        # ----------- FWD appendix queues -----------
        # Attention mask queue (tensors in attention_mask_from_pre_rank_queue is same as attention_mask_to_next_rank_queue)
        self.attention_mask_from_pre_rank_queue: Queue[Tuple[int, int, torch.Tensor]] = Queue(maxsize=self.num_micro_batches)
        self.attention_mask_to_next_rank_queue: Queue[Tuple[int, int, torch.Tensor]] = Queue(maxsize=self.num_micro_batches)
        # Position embeddings queue (Model with ROPE)
        # self.position_embeddings_queue: Queue[Tuple[int, int, Tuple[torch.Tensor, torch.Tensor]]] = Queue()

    def _init_pipeline_info(self):
        # ---- Pipeline info
        self.global_batch_idx = 0
        self.pipeline_mode = self.server_args.pipeline_mode
        self.batch_size = self.server_args.batch_size
        self.micro_batch_size = self.server_args.micro_batch_size
        self.num_micro_batches = (self.batch_size + self.micro_batch_size - 1) // self.micro_batch_size
        # TODO 不同流水线可能得不同地使用
        self.fwd_mb_count = 0
        self.bwd_mb_count = 0

    def _is_last_mb_of_global_batch(self) -> bool:
        return self.fwd_mb_count == self.num_micro_batches - 1

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
            self._compute_task()
        except Exception as e:
            self.logger.exception(f"Compute loop crashed: {e}")
        finally:
            print(f"Rank {self.rank} Compute loop exited")
            self.shutdown()

    @abstractmethod
    def _recv_loop(self):
        raise NotImplementedError('Method not implemented')

    @abstractmethod
    def _send_loop(self):
        raise NotImplementedError('Method not implemented')

    @abstractmethod
    def _compute_task(self):
        raise NotImplementedError('Method not implemented')

    def shutdown(self):
        """Graceful shutdown for futures, sockets, and executor."""
        if not self.stop_event.is_set():
            print(f"Rank {self.rank} Shutting down server...")
            # 1. set stop event to signal loops to exit
            self.stop_event.set()
            # 2. wait for loops to exit
            # self.client_future.result()
            # self.server_future.result()
            # self.compute_future.result()
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
        return -1 if self.rank <= 0 else self.rank - 1

    @property
    def recv_bwd_grad_rank(self) -> int:
        return -1 if self.rank >= self.world_size - 1 else self.rank + 1

    @property
    def send_fwd_acti_rank(self) -> int:
        return self.recv_bwd_grad_rank  # same as recv_bwd_grad_rank

    @property
    def send_bwd_grad_rank(self) -> int:
        return self.recv_fwd_acti_rank  # same as recv_fwd_acti_rank

    def _send_tensor(
        self,
        dst_rank: int,
        retrive_queue: Queue[Tuple[int, int, torch.Tensor]],
        block: bool = False,
    ):
        """
        从对应队列里取出一个 (msg_type, tensor) 并发送, 由发送线程做
        retrive_queue 里的元素形如: (msg_type, tensor)
        Header 格式:
        [
            msg_type,         # 消息类型(1/2/3/N)
            microbatch_idx,   # 当前流水线中 microbatch ID
            ndim,             # 真实维度（几维）
            dim0, dim1, ...   # 各维度大小
            0, 0, ...         # padding
        ]
        """
        if dst_rank == -1:
            self._send_socket_msg(retrive_queue, block=block)
        else:
            self._send_nccl_msg(dst_rank, retrive_queue, block=block)
        pass

    def _put_recv_tensor_to_queue(self, msg_type, microbatch_idx, tensor, block=False):
        # print(f'Rank {self.rank} put tensor to queue, msg_type {msg_type}, mb_idx {microbatch_idx}')
        if msg_type == MSG_TYPE_ACTIVATION:
            self.activation_from_pre_rank_queue.put((msg_type, microbatch_idx, tensor), block=block)
        elif msg_type == MSG_TYPE_GRADIENT:
            self.gradient_from_next_rank_queue.put((msg_type, microbatch_idx, tensor), block=block)
        elif msg_type == MSG_TYPE_ATTENTION_MASK:
            self.attention_mask_from_pre_rank_queue.put((msg_type, microbatch_idx, tensor), block=block)
        else:
            raise ValueError(f"Unknown msg_type: {msg_type}")
        print(f'Rank {self.rank} put tensor to queue, msg_type {msg_type}, mb_idx {microbatch_idx}')

    # def _put_tensor_to_queue(self, msg_type, microbatch_idx, tensor, block=False):
    #     print(f'put tensor to queue, msg_type {msg_type}, mb_idx {microbatch_idx}')
    #     if msg_type == MSG_TYPE_ACTIVATION:
    #         self.activation_from_pre_rank_queue.put((msg_type, microbatch_idx, tensor), block=block)
    #     elif msg_type == MSG_TYPE_GRADIENT:
    #         self.gradient_to_pre_rank_queue.put((msg_type, microbatch_idx, tensor), block=block)
    #     elif msg_type == MSG_TYPE_ATTENTION_MASK:
    #         self.attention_mask_queue.put((msg_type, microbatch_idx, tensor), block=block)
    #     else:
    #         raise ValueError(f"Unknown msg_type: {msg_type}")

    def _recv_mb_fwd_tensors(self, dtype: torch.dtype, block: bool = False):
        """
        接收前向传播的激活张量和attention mask
        """
        # recv activation
        # print(f"Rank {self.rank} try to recv activation from {self.recv_fwd_acti_rank}")
        self._recv_tensor(self.recv_fwd_acti_rank, dtype, block=block)
        if self.recv_fwd_acti_rank != -1:
            # recv attention mask
            self._recv_tensor(self.recv_fwd_acti_rank, dtype, block=block)

    def _recv_mb_bwd_grads(self, dtype: torch.dtype, block: bool = False):
        """
        接收反向传播的梯度张量
        """
        # recv gradient
        self._recv_tensor(self.recv_bwd_grad_rank, dtype, block=block)

    def _send_mb_fwd_tensors(self, block: bool = False):
        """
        发送前向传播的激活张量和attention mask
        """
        # send activation
        self._send_tensor(self.send_fwd_acti_rank, self.activation_from_pre_rank_queue, block=block)
        if self.send_fwd_acti_rank != -1:
            # send attention mask
            self._send_tensor(self.send_fwd_acti_rank, self.attention_mask_to_next_rank_queue, block=block)

    def _send_mb_bwd_grads(self, block: bool = False):
        """
        发送反向传播的梯度张量
        """
        # send gradient
        self._send_tensor(self.send_bwd_grad_rank, self.gradient_from_next_rank_queue, block=block)

    @torch.no_grad()
    def _recv_tensor(self, src_rank: int, dtype: torch.dtype, block: bool = False):

        if src_rank == -1:  # 从 socket 接收
            # print(f'Rank {self.rank} try to recv tensor from socket, sock={self.communicator.conn}')
            payload = self.communicator.receive()

            if payload is None:
                print(f"Rank {self.rank} receive None (peer closed?)")
                return

            if isinstance(payload, dict) and "stop" in payload:
                print("Client requested stop")
                # TODO: 标记 stop，或者放一个特殊消息进队列
                return

            payload: Payload
            msg_type = MSG_TYPE_ACTIVATION if payload.is_activation else MSG_TYPE_GRADIENT
            mb_idx = payload.mb_idx
            print(f"Rank {self.rank} Received tensor, msg_type {msg_type}, mb_idx {mb_idx}, load to gpu {self.device}")

            tensor = payload.tensor.to(self.device, dtype=dtype)
            self._put_recv_tensor_to_queue(msg_type, mb_idx, tensor, block)

            if msg_type == MSG_TYPE_ACTIVATION:
                attn = payload.attention_mask.to(self.device, dtype=dtype) if payload.attention_mask is not None else None
                self._put_recv_tensor_to_queue(MSG_TYPE_ATTENTION_MASK, mb_idx, attn, block=block)

        else:
            msg = self._recv_nccl_msg(src_rank, dtype)
            # 放到对应队列中
            self._put_recv_tensor_to_queue(*msg, block=block)

    # def _recv_socket_msg(self) -> Union[Payload, Dict]:
    #     """
    #     接收 socket 消息
    #     """
    #     self.communicator.sock.settimeout(60.0)

    #     return data

    @torch.no_grad()
    def _recv_nccl_msg(self, src_rank: int, dtype: torch.dtype) -> torch.Tensor:
        """
        接收一个张量并放到指定队列,由接收线程做
        接收到的会以 (msg_type, tensor) 的形式放入 placement_queue
        """
        header_len = MAX_DIM + 3

        # ----- 1) 接收 header -----
        header_tensor: torch.Tensor = torch.empty(header_len, dtype=torch.int64, device=self.device)
        dist.recv(header_tensor, src=src_rank, tag=0)

        header_list = header_tensor.tolist()
        msg_type = int(header_list[0])
        microbatch_idx = int(header_list[1])
        ndim = int(header_list[2])
        dims = header_list[3 : 3 + ndim]
        real_shape = [int(d) for d in dims]

        # ----- 2) 接收 flat tensor -----
        numel = int(torch.prod(torch.tensor(real_shape)))
        flat_tensor: torch.Tensor = torch.empty(numel, dtype=dtype, device=self.device)
        dist.recv(flat_tensor, src=src_rank, tag=1)

        tensor = flat_tensor.view(*real_shape)
        print(f"Rank {self.rank} Received tensor, msg_type {msg_type}, mb_idx {microbatch_idx}")
        return (msg_type, microbatch_idx, tensor)

    @torch.no_grad()
    def _send_socket_msg(self, retrive_queue: Queue[Tuple[int, int, torch.Tensor]], block: bool = False):
        """
        发送 socket 消息
        """
        if retrive_queue == self.attention_mask_to_next_rank_queue:
            # 不用发送attention mask到 client 端
            return
        msg_type, microbatch_idx, tensor = retrive_queue.get_nowait()
        if tensor.is_cuda:
            cpu_tensor = tensor.cpu().pin_memory()
        # send CPU payload to client
        payload = Payload(
            tensor=cpu_tensor,
            is_activation=msg_type == MSG_TYPE_ACTIVATION,
            phase='FWD' if msg_type == MSG_TYPE_ACTIVATION else 'BWD',
            mb_idx=microbatch_idx,
            mb_total=self.num_micro_batches,
            attention_mask=None,
            position_embeddings=None,
            # TODO solve token and group_id in Payload
        )
        self.communicator.send(payload)
        pass

    @torch.no_grad()
    def _send_nccl_msg(self, dst_rank: int, retrive_queue: Queue[Tuple[int, int, torch.Tensor]], block: bool = False):
        """
        发送 NCCL 消息
        """
        msg_type, microbatch_idx, tensor = retrive_queue.get_nowait()

        assert tensor.is_cuda, "NCCL 要求 tensor 在 GPU 上"
        shape = list(tensor.shape)
        ndim = len(shape)
        assert ndim <= MAX_DIM, f"tensor 维度 {ndim} > MAX_DIM={MAX_DIM}"

        # --- 1. 发送 header: [msg_type, ndim, dim0, dim1, ..., 0, 0] ---
        header_len = MAX_DIM + 3
        header = [msg_type, microbatch_idx, ndim] + shape
        header += [0] * (header_len - len(header))

        header_tensor = torch.tensor(header, dtype=torch.int64, device=tensor.device)
        wk1 = dist.isend(header_tensor, dst=dst_rank, tag=0)  # 异步发送header

        # --- 2. 发送数据本体（flatten） ---
        flat_tensor = tensor.contiguous().view(-1)
        wk2 = dist.isend(flat_tensor, dst=dst_rank, tag=1)
        wk1.wait()
        wk2.wait()
        pass

    def _update_stage(self):
        # update model
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
