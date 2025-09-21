from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
import socket
import threading
import logging
import time
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn

from usl.socket import SocketCommunicator
from usl.utils.exp import fed_avg_params


# -------------------------------
# Args
# -------------------------------
@dataclass
class ServerArgs:
    port: int = 8000
    step: int = 10
    use_lora: bool = True
    model: str = "meta-llama/llama3.2-1b"
    server_device: str = "cuda:0"
    split_point: int = 2
    dataset: str = "gsm8k"
    learning_rate: float = 5e-4
    # NOTE: original had a typo 'rete_limit_mbps'. Kept for backward-compat, but also expose the correct name.
    rete_limit_mbps: float = 10
    rate_limit_mbps: Optional[float] = None

    def effective_rate_limit(self) -> float:
        # Prefer the correctly spelled one if provided
        return self.rate_limit_mbps if self.rate_limit_mbps is not None else self.rete_limit_mbps


# -------------------------------
# Base class with ThreadPoolExecutor
# -------------------------------
class ServerBase:
    """Server base that uses ThreadPoolExecutor instead of raw Threads.

    Responsibilities:
      - Own a single accept loop (server supports a single client at a time).
      - Run compute & client-comm loops via executor workers.
      - Provide graceful shutdown & robust exception logging.
    """

    def __init__(self, server_args: ServerArgs, logger: logging.Logger):
        super().__init__()
        self.server_args = server_args
        self.server_device = server_args.server_device
        self.logger: logging.Logger = logger

        # Communicator
        self.communicator = SocketCommunicator(
            is_server=True,
            port=server_args.port,
            buffer_size=1024 * 1024,  # 1MB
            rate_limit_mbps=server_args.effective_rate_limit(),
        )

        # Single client
        self.conn: Optional[socket.socket] = None
        self.addr: Optional[Tuple] = None

        # Metrics
        self.compute_time = 0.0
        self.aggregate_server_time = 0.0
        self.aggregate_client_time = 0.0
        self.idle_time = 0.0

        # Executors & coordination
        # - main_executor runs long-lived loops (compute & client-comm)
        #   Using max_workers=2 since we only need those two loops concurrently.
        self.main_executor: Optional[ThreadPoolExecutor] = None
        self.compute_future: Optional[Future] = None
        self.client_future: Optional[Future] = None
        self.stop_event = threading.Event()

        # Queues (compute pipeline)
        self.activation_from_head_queue: Queue = Queue()
        self.activation_to_tail_queue: Queue = Queue()
        self.gradient_from_tail_queue: Queue = Queue()
        self.gradient_to_head_queue: Queue = Queue()

        # Micro-batch context/state
        self.ctx: Dict[str, Dict[str, torch.Tensor]] = {}
        self.group_state: Dict[str, Dict[str, Any]] = {}

    # ---------------- Abstracts ----------------
    def _forward(
        self,
        activation: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings=None,
        *,
        token: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _backward(self, server_grad: torch.Tensor, *, token: str) -> torch.Tensor:
        raise NotImplementedError

    def handle_client_request(self):
        raise NotImplementedError

    def compute_task(self):
        raise NotImplementedError

    # ---------------- Lifecycle ----------------
    def run(self):
        """Start compute loop and accept a single client, all via executor workers."""
        self.main_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="server")

        # Start compute loop immediately
        self.compute_future = self.main_executor.submit(self._compute_loop_wrapper)

        # Accept a single client (blocking here, but we're okay because compute loop runs on executor)
        while not self.stop_event.is_set():
            conn, addr = self.communicator.accept_client()
            if conn is None:
                # accept_client returns (None, None) on timeout; keep waiting unless stopping
                continue
            self.conn = conn
            self.addr = addr
            self.logger.info(f"Connected from {addr}")
            # Start client communication loop on executor
            self.client_future = self.main_executor.submit(self._client_loop_wrapper)
            break

        self.logger.info("Client connected")

    def shutdown(self, wait: bool = True):
        """Graceful shutdown for loops, sockets, and executor."""
        self.stop_event.set()
        try:
            self.communicator.close()
            if self.conn:
                try:
                    self.conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.conn.close()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

        if self.main_executor:
            self.main_executor.shutdown(wait=wait, cancel_futures=True)

        self.conn = None
        self.addr = None

    # ---------------- Internal wrappers with robust logging ----------------
    def _client_loop_wrapper(self):
        try:
            self.handle_client_request()
        except Exception as e:
            self.logger.exception(f"Client loop crashed: {e}")
            self.stop_event.set()

    def _compute_loop_wrapper(self):
        try:
            self.compute_task()
        except Exception as e:
            self.logger.exception(f"Compute loop crashed: {e}")
            self.stop_event.set()

    # ---------------- Communication helper ----------------
    def _send_to_client(self, response: Dict) -> bool:
        if not self.conn:
            self.logger.error("No active client connection")
            return False
        try:
            self.communicator.send(response)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message to client (addr={self.addr}): {e}")
            return False
