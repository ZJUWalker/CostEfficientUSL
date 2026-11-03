"""Client head/tail training primitives and Ray actor."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import ray
import torch
import torch.nn.functional as functional
from torch import nn

from usl_ray.client.schedules import SCHEDULES
from usl_ray.config import ExperimentConfig
from usl_ray.data import Batch, create_dataloader, exactly_steps
from usl_ray.models.loader import load_client_partitions, load_tokenizer
from usl_ray.models.sequence import MaskSpec, PositionSpec, SequenceSpec
from usl_ray.offload import AsyncActivationOffloader, ModelStateSwapper
from usl_ray.profile.recorder import Recorder, merge_timelines
from usl_ray.profile.timeline import StepTimeline
from usl_ray.server.stage_actor import ServerInfo, StageResult
from usl_ray.transport import Mailbox, MessageKey, Outbox, TensorBlob, clock_ms


@dataclass(frozen=True)
class MicroBatch:
    """One client-owned microbatch."""

    index: int
    input_ids: torch.Tensor
    labels: torch.Tensor
    sequence: SequenceSpec
    loss_divisor: int


@dataclass(frozen=True)
class ClientResult:
    """Minimal result extended with profile spans in M8."""

    losses: tuple[float, ...]
    max_memory_mib: float
    timelines: tuple[StepTimeline, ...]
    server_memory_mib: tuple[float, ...]


class ClientTrainer:
    """Stateful client primitives consumed by simple schedule functions."""

    def __init__(
        self,
        config: ExperimentConfig,
        head: nn.Module,
        tail: nn.Module,
        server_stage_zero: ray.actor.ActorHandle,
        server_stage_last: ray.actor.ActorHandle,
        mailbox: Mailbox,
        device: torch.device,
        server_world_size: int,
        recorder: Recorder,
    ) -> None:
        self.config = config
        self.head = head.to(device).train()
        self.tail = tail.to(device).train()
        self.server_stage_zero = server_stage_zero
        self.server_stage_last = server_stage_last
        self.mailbox = mailbox
        self.device = device
        self.server_world_size = server_world_size
        self.recorder = recorder
        self.outbox = Outbox(
            device,
            lambda key, start, end, nbytes: recorder.transfer("uplink", key, start, end, nbytes),
        )
        self._receive_executor = ThreadPoolExecutor(
            max_workers=2 * config.batch.num_microbatches,
            thread_name_prefix="usl-receive",
        )
        self._receive_futures: dict[MessageKey, Future[torch.Tensor]] = {}
        self._head_outputs: dict[int, torch.Tensor] = {}
        self._tail_inputs: dict[int, torch.Tensor] = {}
        self._tail_losses: dict[int, torch.Tensor] = {}
        self.current_step = -1
        head_parameters = list(self.head.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        tail_parameters = [parameter for parameter in self.tail.parameters() if id(parameter) not in head_ids]
        self.optimizer_head = torch.optim.AdamW(head_parameters, lr=config.train.learning_rate)
        self.optimizer_tail = torch.optim.AdamW(tail_parameters, lr=config.train.learning_rate)
        self.head_swapper: ModelStateSwapper | None = None
        self.tail_swapper: ModelStateSwapper | None = None
        self.activation_offloader: AsyncActivationOffloader | None = None
        if config.client is not None and device.type == "cuda":
            layer_count = config.client.model_state_offload_layers
            if layer_count:
                tail_ids = {id(parameter) for parameter in self.tail.parameters()}
                shared_ids = {id(parameter) for parameter in head_parameters if id(parameter) in tail_ids}
                self.head_swapper = ModelStateSwapper(
                    self.head,
                    self.optimizer_head,
                    layer_count,
                    device,
                    excluded_parameter_ids=shared_ids,
                )
                self.tail_swapper = ModelStateSwapper(
                    self.tail,
                    self.optimizer_tail,
                    layer_count,
                    device,
                    excluded_parameter_ids=shared_ids,
                )
                self.tail_swapper.offload()
            if config.client.activation_offload_mb or self.head_swapper is not None:
                parameters = self.head_swapper.parameters if self.head_swapper is not None else []
                self.activation_offloader = AsyncActivationOffloader(device, parameters)

    def _receive(self, key: MessageKey) -> torch.Tensor:
        blob = self.mailbox.get(key, self.config.network.timeout_s)
        self.mailbox.discard(key)
        return blob.to_tensor(self.device, non_blocking=self.device.type == "cuda")

    def _future(self, key: MessageKey) -> Future[torch.Tensor]:
        future = self._receive_futures.get(key)
        if future is None:
            future = self._receive_executor.submit(self._receive, key)
            self._receive_futures[key] = future
        return future

    def head_forward(self, microbatch: MicroBatch) -> None:
        """Run head forward and submit its activation to server rank zero."""
        offload_count = self.config.client.activation_offload_mb if self.config.client is not None else 0
        context = (
            self.activation_offloader.context(offload_activations=microbatch.index < offload_count)
            if self.activation_offloader is not None
            else nullcontext()
        )
        with self.recorder.span("client", "head_forward", self.current_step, microbatch.index, self.device):
            with context:
                output = self.head(microbatch.input_ids, microbatch.sequence)
        self._head_outputs[microbatch.index] = output
        self.outbox.submit(
            self.server_stage_zero,
            MessageKey(self.current_step, microbatch.index, "act_up"),
            output,
        )

    def activation(self, microbatch: MicroBatch) -> Future[torch.Tensor]:
        """Return the future for the server activation consumed by the tail."""
        return self._future(MessageKey(self.current_step, microbatch.index, "act_down"))

    def tail_forward(self, microbatch: MicroBatch, activation: object) -> None:
        """Run tail forward and retain its differentiable scalar loss."""
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"Expected activation tensor, got {type(activation)}")
        tail_input = activation.detach().requires_grad_(True)
        with self.recorder.span("client", "tail_forward", self.current_step, microbatch.index, self.device):
            logits = self.tail(tail_input, microbatch.sequence)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.shape[-1])
            shift_labels = microbatch.labels[..., 1:].contiguous().view(-1)
            loss = functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
        self._tail_inputs[microbatch.index] = tail_input
        self._tail_losses[microbatch.index] = loss / microbatch.loss_divisor

    def tail_backward(self, microbatch: MicroBatch) -> float:
        """Backpropagate tail loss and submit the boundary gradient."""
        loss = self._tail_losses.pop(microbatch.index)
        tail_input = self._tail_inputs.pop(microbatch.index)
        with self.recorder.span("client", "tail_backward", self.current_step, microbatch.index, self.device):
            loss.backward()
        if tail_input.grad is None:
            raise RuntimeError(f"Missing tail input gradient for microbatch {microbatch.index}")
        self.outbox.submit(
            self.server_stage_last,
            MessageKey(self.current_step, microbatch.index, "grad_up"),
            tail_input.grad,
        )
        return float(loss.detach())

    def gradient(self, microbatch: MicroBatch) -> Future[torch.Tensor]:
        """Return the future for the gradient consumed by the head."""
        return self._future(MessageKey(self.current_step, microbatch.index, "grad_down"))

    def head_backward(self, microbatch: MicroBatch, gradient: object) -> None:
        """Backpropagate through the head and release its graph."""
        if not isinstance(gradient, torch.Tensor):
            raise TypeError(f"Expected gradient tensor, got {type(gradient)}")
        with self.recorder.span("client", "head_backward", self.current_step, microbatch.index, self.device):
            self._head_outputs.pop(microbatch.index).backward(gradient)

    def swap_to_tail(self) -> None:
        """Hook point for model-state offload implemented in M6."""
        if self.head_swapper is not None and self.tail_swapper is not None:
            with self.recorder.span("client", "head_model_offload", self.current_step, -1, self.device):
                self.head_swapper.offload()
            with self.recorder.span("client", "tail_model_reload", self.current_step, -1, self.device):
                self.tail_swapper.reload()

    def swap_to_head(self) -> None:
        """Hook point for model-state offload implemented in M6."""
        if self.head_swapper is not None and self.tail_swapper is not None:
            with self.recorder.span("client", "tail_model_offload", self.current_step, -1, self.device):
                self.tail_swapper.offload()
            with self.recorder.span("client", "head_model_reload", self.current_step, -1, self.device):
                self.head_swapper.reload()

    def step_tail(self) -> None:
        """Clip and update tail-only parameters."""
        with self.recorder.span("client", "tail_step", self.current_step, -1, self.device):
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in self.optimizer_tail.param_groups for parameter in group["params"]],
                self.config.train.max_grad_norm,
            )
            self.optimizer_tail.step()

    def step_head(self) -> None:
        """Clip and update head parameters, including tied embeddings."""
        with self.recorder.span("client", "head_step", self.current_step, -1, self.device):
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.config.train.max_grad_norm)
            self.optimizer_head.step()

    def run_batch(self, batch: Batch, step: int) -> float:
        """Split one batch, align the global step, and execute its schedule."""
        self.current_step = step
        self.optimizer_head.zero_grad(set_to_none=True)
        self.optimizer_tail.zero_grad(set_to_none=True)
        microbatches = self._split_batch(batch)
        ray.get(self.server_stage_zero.begin_step.remote(step, [item.sequence for item in microbatches]))
        loss = SCHEDULES[self.config.pipeline.mode](self, microbatches)
        self.outbox.flush()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        ray.get(self.server_stage_zero.finish_step.remote(step))
        self._receive_futures.clear()
        return loss

    def _split_batch(self, batch: Batch) -> list[MicroBatch]:
        input_ids = batch.input_ids.to(self.device)
        labels = batch.labels.to(self.device)
        attention_mask = batch.attention_mask
        shifted_valid = int((labels[..., 1:] != -100).sum())
        if shifted_valid <= 0:
            raise ValueError("A batch must contain at least one supervised next-token target")
        inputs = torch.split(input_ids, self.config.batch.micro_batch_size)
        label_chunks = torch.split(labels, self.config.batch.micro_batch_size)
        mask_chunks = torch.split(attention_mask, self.config.batch.micro_batch_size)
        result: list[MicroBatch] = []
        for index, (ids, chunk_labels, mask) in enumerate(zip(inputs, label_chunks, mask_chunks, strict=True)):
            valid_lens: tuple[int, ...] | None = None
            if self.config.batch.mask_kind == "full" or self.config.batch.padding_side == "left":
                valid_lens = tuple(int(item) for item in mask.sum(dim=-1))
            result.append(
                MicroBatch(
                    index=index,
                    input_ids=ids,
                    labels=chunk_labels,
                    sequence=SequenceSpec(
                        seq_len=ids.shape[1],
                        valid_lens=valid_lens,
                        mask=MaskSpec(self.config.batch.mask_kind),
                        position=PositionSpec(self.config.batch.position_kind),
                    ),
                    loss_divisor=shifted_valid,
                )
            )
        return result

    def close(self) -> None:
        """Release communication workers."""
        self.outbox.close()
        self._receive_executor.shutdown(wait=True, cancel_futures=True)
        if self.activation_offloader is not None:
            self.activation_offloader.close()


@ray.remote(concurrency_groups={"compute": 1, "io": 8})
class ClientActor:
    """Own client models, data, optimizers, and the boundary mailbox."""

    def __init__(
        self,
        config: ExperimentConfig,
        server_stage_zero: ray.actor.ActorHandle,
    ) -> None:
        if config.client is None:
            raise ValueError("ClientActor requires client config")
        self.config = config
        self.server_stage_zero = server_stage_zero
        self.server_info: ServerInfo | None = None
        self.mailbox = Mailbox()
        self._aborted = threading.Event()
        self._configured = threading.Event()
        self.head: nn.Module | None = None
        self.tail: nn.Module | None = None
        self.recorder = Recorder(config.profile.enabled)
        self._server_results: Future[list[StageResult]] = Future()

    @ray.method(concurrency_group="io")
    def configure(self, server_info: ServerInfo) -> None:
        """Finish the two-way handshake after the server receives this handle."""
        self.server_info = server_info
        self._configured.set()

    @ray.method(concurrency_group="io")
    def set_server_results(self, results: list[StageResult]) -> None:
        """Receive final timelines and memory before the client driver exits."""
        if not self._server_results.done():
            self._server_results.set_result(results)

    @ray.method(concurrency_group="io")
    def push(self, key: MessageKey, blob: TensorBlob) -> None:
        """Receive a Ray-dereferenced boundary tensor."""
        self.mailbox.put(key, blob)

    @ray.method(concurrency_group="io")
    def abort(self, reason: str) -> None:
        """Unblock training after a server failure."""
        self.mailbox.abort(reason)
        self._aborted.set()

    @ray.method(concurrency_group="io")
    def clock(self) -> float:
        """Return a profile clock sample."""
        return clock_ms()

    @ray.method(concurrency_group="io")
    def node_id(self) -> str:
        """Return the Ray node identifier hosting the client."""
        return ray.get_runtime_context().get_node_id()

    @ray.method(concurrency_group="io")
    def parameter_state(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return CPU snapshots for numerical integration tests."""
        if self.head is None or self.tail is None:
            raise RuntimeError("Client models are not initialized")
        return (
            {name: value.detach().cpu().clone() for name, value in self.head.state_dict().items()},
            {name: value.detach().cpu().clone() for name, value in self.tail.state_dict().items()},
        )

    @ray.method(concurrency_group="compute")
    def train(self) -> ClientResult:
        """Execute exactly the configured global steps."""
        assigned = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        device = torch.device("cuda", 0) if assigned else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.manual_seed(self.config.run.seed)
        head, tail = load_client_partitions(self.config)
        self.head, self.tail = head, tail
        tokenizer = (
            None
            if self.config.client is None or self.config.client.data.dataset == "synthetic"
            else load_tokenizer(self.config)
        )
        loader = create_dataloader(self.config, tokenizer)
        if not self._configured.wait(timeout=self.config.client.wait_server_s):
            raise TimeoutError("Client timed out waiting for server handshake")
        if self.server_info is None:
            raise RuntimeError("Server handshake did not provide ServerInfo")
        trainer = ClientTrainer(
            self.config,
            head,
            tail,
            self.server_stage_zero,
            self.server_info.last_stage,
            self.mailbox,
            device,
            self.server_info.world_size,
            self.recorder,
        )
        losses: list[float] = []
        try:
            for step, batch in enumerate(exactly_steps(loader, self.config.train.steps)):
                losses.append(trainer.run_batch(batch, step))
            memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            server_results = self._server_results.result(timeout=self.config.network.timeout_s)
            timelines = merge_timelines(
                self.recorder.timelines(),
                *(item.timelines for item in server_results),
            )
            return ClientResult(
                tuple(losses),
                memory,
                timelines,
                tuple(item.max_memory_mib for item in server_results),
            )
        except BaseException as error:
            self.server_stage_zero.abort.remote(f"Client failed: {error}")
            raise
        finally:
            trainer.close()
