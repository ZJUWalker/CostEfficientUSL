from __future__ import annotations

from dataclasses import dataclass
from queue import Empty
import socket
import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from usl.server.server_base import ServerArgs, ServerBase


# -------------------------------
# SingleServer implementation
# -------------------------------
class SingleServer(ServerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        server_model: nn.Module,
        optimizer_clz: type[torch.optim.Optimizer] = torch.optim.AdamW,
        logger: Optional[logging.Logger] = None,
        matrix_logger: Optional[logging.Logger] = None,
    ):
        super().__init__(server_args=server_args, logger=logger)
        self.lr = server_args.learning_rate
        self.trunk_model = server_model.to(self.server_device).train()
        self.optimizer: torch.optim.Optimizer = optimizer_clz(self.trunk_model.parameters(), lr=self.lr)
        self.matrix_logger = matrix_logger
        self.server_output: Optional[torch.Tensor] = None
        self.hidden_status_from_head: Optional[torch.Tensor] = None

    # -------- Forward / Backward --------
    def _forward(
        self,
        activation: torch.Tensor,
        attention_mask: torch.LongTensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        token: str,
    ) -> torch.Tensor:
        try:
            hidden_status_from_head = activation.to(self.server_device)
            hidden_status_from_head.requires_grad_(True)
            # hidden_status_from_head.retain_grad()

            attention_mask = attention_mask.to(self.server_device) if attention_mask is not None else None
            pos_emb = (
                tuple(pos.to(self.server_device) for pos in position_embeddings)
                if position_embeddings is not None
                else None
            )

            fwd_start = time.time()
            server_output: torch.Tensor = self.trunk_model(
                hidden_states=hidden_status_from_head,
                attention_mask=attention_mask,
                position_embeddings=pos_emb,
            )
            if torch.cuda.is_available() and str(self.server_device).startswith("cuda"):
                torch.cuda.synchronize(device=self.server_device)
            self.compute_time += time.time() - fwd_start

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

    def _backward(self, server_grad: torch.Tensor, *, token: str) -> torch.Tensor:
        try:
            ctx = self.ctx.pop(token, None)
            if ctx is None:
                raise RuntimeError(f"Missing context for token={token}")

            server_output = ctx["server_output"]
            hidden_from_head = ctx["hidden_from_head"]

            server_grad = server_grad.to(self.server_device)

            bwd_start_time = time.time()
            server_output.backward(server_grad, retain_graph=False)

            grad_to_head = hidden_from_head.grad.cpu()

            if torch.cuda.is_available() and str(self.server_device).startswith("cuda"):
                torch.cuda.synchronize(device=self.server_device)
            self.compute_time += time.time() - bwd_start_time
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return grad_to_head
        except Exception as e:
            self.logger.error(f"Backward failed (token={token}): {e}")
            raise

    # -------- Client loop (now executor-run) --------
    def handle_client_request(self):
        client_id = None
        try:
            if not self.conn:
                # Wait for run() to set a connection; this path is rarely hit.
                while not self.stop_event.is_set() and not self.conn:
                    time.sleep(0.05)
            if not self.conn:
                return

            self.conn.settimeout(60.0)
            while not self.stop_event.is_set():

                data: Dict = self.communicator.receive()
                if data is None:
                    self.logger.info(f"Client {self.addr} disconnected")
                    break

                if "activation" in data:
                    self.activation_from_head_queue.put(data)
                    # Wait for compute to produce a response
                    try:
                        response: Dict = self.activation_to_tail_queue.get(timeout=60.0)
                    except Empty:
                        self.logger.error("Timeout waiting for activation response from compute loop")
                        break

                    for k in ("token", "group_id", "micro_idx", "micro_total"):
                        response.setdefault(k, data.get(k))
                    self._send_to_client(response)

                elif "gradient" in data:
                    self.gradient_from_tail_queue.put(data)
                    try:
                        response: Dict = self.gradient_to_head_queue.get(timeout=60.0)
                    except Empty:
                        self.logger.error("Timeout waiting for gradient response from compute loop")
                        break

                    for k in ("token", "group_id", "micro_idx", "micro_total"):
                        response.setdefault(k, data.get(k))
                    self._send_to_client(response)

        except Exception as e:
            self.logger.error(f"Client {self.addr} (client_id={client_id}) error: {e}")
        finally:
            try:
                self.communicator.close()
                if self.conn:
                    try:
                        self.conn.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    self.conn.close()
            finally:
                self.conn = None
                self.addr = None
                self.stop_event.set()

    # -------- Compute loop (executor-run) --------
    def compute_task(self):
        if torch.cuda.is_available() and str(self.server_device).startswith("cuda"):
            dev = torch.device(self.server_device)
            torch.cuda.set_device(dev.index or 0)  # 绑定当前线程的设备
            # 触发一次轻量 CUDA 调用，强制创建上下文（可选但很有用）
            torch.empty(0, device=dev)
        """Non-busy compute loop using queue.get(timeout) for backpressure."""
        while not self.stop_event.is_set():
            # --- Forward path ---
            try:
                fwd_wait_start = time.time()
                data: Dict = self.activation_from_head_queue.get(timeout=0.05)
                self.idle_time += time.time() - fwd_wait_start

                token = str(data.get("token"))
                group_id = str(data.get("group_id", "default"))
                micro_total = int(data.get("micro_total", 1))
                micro_idx = int(data.get("micro_idx", 0))

                # Group init / maintenance
                gs = self.group_state.get(group_id)
                if gs is None:
                    self.group_state[group_id] = {"total": micro_total, "done": 0, "zeroed": True}
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    if data.get("micro_total") is not None:
                        self.group_state[group_id]["total"] = int(data["micro_total"])  # tolerate late info
                # print(f"Server fwd for micro_total={micro_total}, batch_idx={micro_idx}")
                server_activation = self._forward(
                    data["activation"],
                    data.get("attention_mask"),
                    data.get("position_embeddings"),
                    token=token,
                )
                self.activation_to_tail_queue.put(
                    {
                        "server_activation": server_activation,
                        "token": token,
                        "group_id": group_id,
                        "micro_total": micro_total,
                    }
                )
            except Empty:
                pass

            # --- Backward path ---
            try:
                bwd_wait_start = time.time()
                data = self.gradient_from_tail_queue.get(timeout=0.01)
                self.idle_time += time.time() - bwd_wait_start

                token = str(data.get("token"))
                group_id = str(data.get("group_id", "default"))
                micro_idx = int(data.get("micro_idx", 0))
                # print(f"Server bwd for micro_total={micro_total}, batch_idx={micro_idx}")
                grad_to_client = self._backward(data["gradient"], token=token)

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
                    {
                        "server_gradient": grad_to_client,
                        "token": token,
                        "group_id": group_id,
                        "micro_idx": micro_idx,
                        "micro_total": total,
                        "group_done": done,
                    }
                )
            except Empty:
                pass

            # Small cooperative yield
            time.sleep(0.001)
