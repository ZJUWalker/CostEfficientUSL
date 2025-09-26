from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import logging
import time
from typing import Dict, Any, Optional, List, Tuple as TTuple, Type, Union

import torch
import torch.nn as nn

from usl.socket import SocketCommunicator, Payload
from usl.utils.usl_gantt_plot import GanttChartData


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
    rate_limit_mbps: float = 10
    rate_limit_mbps: Optional[float] = None

    def effective_rate_limit(self) -> float:
        # Prefer the correctly spelled one if provided
        return self.rate_limit_mbps if self.rate_limit_mbps is not None else self.rate_limit_mbps


# -------------------------------
# Single, non-inheritance server
# -------------------------------
class SingleServer:
    """A self-contained server (no inheritance) that:
    - Accepts a single client over sockets via SocketCommunicator
    - Runs compute & client-comm loops on a ThreadPoolExecutor
    - Streams activations/gradients and performs forward/backward on a trunk model
    """

    def __init__(
        self,
        server_args: ServerArgs,
        server_model: nn.Module,
        optimizer_clz: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        logger: Optional[logging.Logger] = None,
        matrix_logger: Optional[logging.Logger] = None,
    ):
        # ---- Config & logging
        self.server_args = server_args
        self.server_device = server_args.server_device
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.matrix_logger = matrix_logger

        # ---- Communicator
        self.communicator = SocketCommunicator(
            is_server=True,
            port=server_args.port,
            buffer_size=1024 * 1024,  # 1MB
            rate_limit_mbps=server_args.effective_rate_limit(),
        )

        # ---- Metrics
        self.compute_time = 0.0
        self.idle_time = 0.0
        self.profile_data: List[GanttChartData] = []

        # ---- Executors & coordination
        self.main_executor: Optional[ThreadPoolExecutor] = None
        self.compute_future: Optional[Future] = None
        self.client_future: Optional[Future] = None
        self.stop_event = threading.Event()

        # ---- Queues (compute pipeline)
        self.activation_from_head_queue: Queue = Queue()
        self.activation_to_tail_queue: Queue = Queue()
        self.gradient_from_tail_queue: Queue = Queue()
        self.gradient_to_head_queue: Queue = Queue()

        # ---- Micro-batch context/state
        self.ctx: Dict[str, Dict[str, torch.Tensor]] = {}
        self.group_state: Dict[str, Dict[str, Any]] = {}

        # ---- Model & optimizer
        self.lr = server_args.learning_rate
        self.trunk_model = server_model.to(self.server_device).train()
        self.optimizer: torch.optim.Optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)
        self.server_output: Optional[torch.Tensor] = None
        self.hidden_status_from_head: Optional[torch.Tensor] = None

    # --------------- Public lifecycle ---------------
    def run(self):
        """Start compute loop and accept a single client, all via executor workers."""
        self.main_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="server")
        # Start compute loop immediately
        self.compute_future = self.main_executor.submit(self._compute_loop_wrapper)

        # Accept a single client (blocking here; compute loop runs on executor)
        while not self.stop_event.is_set():
            self.communicator.accept_client()
            if self.communicator.conn is None:
                # accept_client returns (None, None) on timeout; keep waiting unless stopping
                continue
            self.logger.info(f"Connected from {self.communicator.conn}")
            # Start client communication loop on executor
            self.client_future = self.main_executor.submit(self._client_send_loop_wrapper)
            self.server_future = self.main_executor.submit(self._server_send_loop_wrapper)
            break

        self.logger.info("Client connected")

    def shutdown(self):
        """Graceful shutdown for loops, sockets, and executor."""
        if not self.stop_event.is_set():
            print("Shutting down server...")
            self.stop_event.set()
            try:
                self.communicator.close()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")

            if self.main_executor:
                self.main_executor.shutdown(wait=True, cancel_futures=True)
            # save_gantt_chart_data(self.profile_data, "log/profile/server.json")

    # --------------- Internal wrappers (robust logging) ---------------
    def _client_send_loop_wrapper(self):
        try:
            self._handle_client_send()
        except Exception as e:
            self.logger.exception(f"Client loop crashed: {e}")
            self.shutdown()

    def _server_send_loop_wrapper(self):
        try:
            self._handle_server_send()
        except Exception as e:
            self.logger.exception(f"Server loop crashed: {e}")
            self.shutdown()

    def _compute_loop_wrapper(self):
        try:
            self._compute_task()
        except Exception as e:
            self.logger.exception(f"Compute loop crashed: {e}")
            self.shutdown()

    # --------------- Forward / Backward ---------------
    def _forward(
        self,
        activation: torch.Tensor,
        attention_mask: Optional[torch.LongTensor],
        position_embeddings: Optional[TTuple[torch.Tensor, torch.Tensor]] = None,
        *,
        token: str,
        mb_idx: int,
    ) -> torch.Tensor:
        try:
            self.profile_data[mb_idx].server_fwd_timestamp[0] = time.time()
            hidden_status_from_head = activation.to(self.server_device)
            hidden_status_from_head.requires_grad_(True)
            # hidden_status_from_head.retain_grad()

            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = tuple(pos.to(self.server_device) for pos in position_embeddings) if position_embeddings is not None else None

            fwd_start = time.time()
            server_output: torch.Tensor = self.trunk_model(
                hidden_states=hidden_status_from_head,
                attention_mask=attention_mask,
                position_embeddings=pos_emb,
            )
            if torch.cuda.is_available() and str(self.server_device).startswith("cuda"):
                torch.cuda.synchronize(device=self.server_device)
            self.compute_time += time.time() - fwd_start
            self.profile_data[mb_idx].server_fwd_timestamp[1] = time.time()
            # Save context per token for later backward
            server_output = server_output.requires_grad_(True)
            self.ctx[token] = {
                "server_output": server_output,
                "hidden_from_head": hidden_status_from_head,
            }

            activation_to_tail = server_output.detach().cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return activation_to_tail
        except Exception as e:
            self.logger.error(f"Forward failed (token={token}): {e}")
            raise

    def _backward(self, server_grad: torch.Tensor, *, token: str, mb_idx: int) -> torch.Tensor:
        try:
            ctx = self.ctx.pop(token, None)
            if ctx is None:
                raise RuntimeError(f"Missing context for token={token}")

            server_output = ctx["server_output"]
            hidden_from_head = ctx["hidden_from_head"]
            self.profile_data[mb_idx].server_bwd_timestamp[0] = time.time()
            server_grad = server_grad.to(self.server_device)

            bwd_start_time = time.time()
            server_output.backward(server_grad, retain_graph=False)

            grad_to_head = hidden_from_head.grad.cpu()

            if torch.cuda.is_available() and str(self.server_device).startswith("cuda"):
                torch.cuda.synchronize(device=self.server_device)
            self.compute_time += time.time() - bwd_start_time
            self.profile_data[mb_idx].server_bwd_timestamp[1] = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return grad_to_head
        except Exception as e:
            self.logger.error(f"Backward failed (token={token}): {e}")
            raise

    def _handle_server_send(self):
        try:
            if not self.communicator.conn:
                # Wait for run() to set a connection; this path is rarely hit.
                while not self.stop_event.is_set() and not self.communicator.conn:
                    time.sleep(0.05)
            if not self.communicator.conn:
                return
            while not self.stop_event.is_set():
                try:
                    response: Payload = self.activation_to_tail_queue.get(timeout=0.01)
                    if response is not None:
                        start_send_time = time.time()
                        self.communicator.send(response)
                        end_send_time = time.time()
                        self.profile_data[response.mb_idx].server_fwd_send_timestamp[0] = start_send_time
                        self.profile_data[response.mb_idx].server_fwd_send_timestamp[1] = end_send_time
                    else:
                        continue
                except Empty:
                    pass
                try:
                    response: Payload = self.gradient_to_head_queue.get(timeout=0.01)
                    if response is not None:  # 可能是 None（队列空）
                        start_send_time = time.time()
                        self.communicator.send(response)
                        end_send_time = time.time()
                        self.profile_data[response.mb_idx].server_bwd_send_timestamp[0] = start_send_time
                        self.profile_data[response.mb_idx].server_bwd_send_timestamp[1] = end_send_time
                    else:
                        continue
                except Empty:
                    pass
            print('_handle_server_send finished')
        except Exception as e:
            self.logger.error(f"Client {self.communicator.addr} error: {e}")
        finally:
            self.shutdown()
        pass

    # --------------- Client loop ---------------
    def _handle_client_send(self):
        try:
            if not self.communicator.conn:
                # Wait for run() to set a connection; this path is rarely hit.
                while not self.stop_event.is_set() and not self.communicator.conn:
                    time.sleep(0.05)
            if not self.communicator.conn:
                return

            self.communicator.conn.settimeout(60.0)
            while not self.stop_event.is_set():

                data: Union[Payload, Dict] = self.communicator.receive()
                if data is None:
                    self.logger.info(f"Client {self.communicator.addr} disconnected")
                    break
                if isinstance(data, dict) and 'stop' in data.keys():
                    print('Client requested stop')
                    self.communicator.send({'profile': self.profile_data})
                    self.logger.info(f"Client {self.communicator.addr} requested profile data and finished training")
                    break
                if data.is_activation:
                    self.activation_from_head_queue.put(data)
                else:
                    self.gradient_from_tail_queue.put(data)
            print('_handle_client_send finished')

        except Exception as e:
            self.logger.error(f"Client {self.communicator.addr} error: {e}")
        finally:
            self.shutdown()

    # --------------- Compute loop ---------------
    def _compute_task(self):
        # 绑定 CUDA 设备并预热上下文（如果可用）
        if torch.cuda.is_available() and str(self.server_device).startswith("cuda"):
            dev = torch.device(self.server_device)
            torch.cuda.set_device(dev.index or 0)  # 绑定当前线程的设备
            torch.empty(0, device=dev)  # 触发一次轻量 CUDA 调用

        """Non-busy compute loop using queue.get(timeout) for backpressure."""
        while not self.stop_event.is_set():
            # --- Forward path ---
            has_fwd = False
            try:
                fwd_wait_start = time.time()
                data: Optional[Dict | Payload] = self.activation_from_head_queue.get_nowait()
                if isinstance(data, dict) and 'stop' in data.keys():
                    print('Server requested stop')
                    break
                self.idle_time += time.time() - fwd_wait_start

                token = data.token
                group_id = data.group_id
                mb_total = data.mb_total
                mb_idx = data.mb_idx
                # group_id = str(data.get("group_id", "default"))
                # mb_total = int(data.get("mb_total", 1))
                # mb_idx = int(data.get("mb_idx", 0))
                if self.profile_data == []:
                    self.profile_data = [GanttChartData(mini_batch_idx=i) for i in range(mb_total)]
                # Group init / maintenance
                gs = self.group_state.get(group_id)
                if gs is None:
                    self.group_state[group_id] = {"total": mb_total, "done": 0, "zeroed": True}
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    if mb_total is not None:
                        # tolerate late info
                        self.group_state[group_id]["total"] = int(data.mb_total)

                server_activation = self._forward(
                    data.tensor,
                    data.attention_mask,
                    data.position_embeddings,
                    token=token,
                    mb_idx=mb_idx,
                )
                self.activation_to_tail_queue.put(
                    Payload(
                        tensor=server_activation,
                        token=token,
                        group_id=group_id,
                        mb_idx=mb_idx,
                        mb_total=mb_total,
                        is_activation=True,
                    )
                )
                has_fwd = True
            except Empty:
                pass

            # Small cooperative yield
            if has_fwd:
                continue

            # --- Backward path ---
            try:

                bwd_wait_start = time.time()
                data = self.gradient_from_tail_queue.get_nowait()
                self.idle_time += time.time() - bwd_wait_start

                token = data.token
                group_id = data.group_id
                mb_idx = data.mb_idx
                # group_id = str(data.get("group_id", "default"))
                # mb_idx = int(data.get("mb_idx", 0))
                grad_to_client = self._backward(data.tensor, token=token, mb_idx=mb_idx)

                gs = self.group_state.setdefault(group_id, {"total": 1, "done": 0, "zeroed": True})
                gs["done"] += 1
                total = int(gs.get("total", 1))
                done = int(gs["done"])

                if done >= total:
                    torch.nn.utils.clip_grad_norm_(self.trunk_model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.group_state.pop(group_id, None)

                self.gradient_to_head_queue.put(
                    # {
                    #     "tensor": grad_to_client,
                    #     "is_activation": False,
                    #     "token": token,
                    #     "group_id": group_id,
                    #     "mb_idx": mb_idx,
                    #     "mb_total": total,
                    #     "group_done": done,
                    # }
                    Payload(
                        tensor=grad_to_client,
                        is_activation=False,
                        token=token,
                        group_id=group_id,
                        mb_idx=mb_idx,
                        mb_total=total,
                    )
                )
            except Empty:
                pass

            # Small cooperative yield
            time.sleep(0.001)
