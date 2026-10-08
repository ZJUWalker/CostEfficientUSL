"""One Ray actor per torch pipeline stage."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray._common.network_utils import find_free_port
from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

from usl_ray.config import ExperimentConfig, ServerConfig, diff_shared_config
from usl_ray.models.loader import load_server_partition
from usl_ray.models.partition import SplitPlan
from usl_ray.models.sequence import SequenceSpec
from usl_ray.offload import AsyncActivationOffloader
from usl_ray.profile.recorder import Recorder
from usl_ray.profile.timeline import StepTimeline
from usl_ray.server.boundary import ClientGradients, ClientInputs, ClientLoss, USLPipelineStage, microbatch_kwargs
from usl_ray.transport import Mailbox, MessageKey, Outbox, TensorBlob, clock_ms


@dataclass(frozen=True)
class ServerInfo:
    """Information returned to the client after a successful attach."""

    world_size: int
    layer_ranges: tuple[tuple[int, int], ...]
    node_id: str
    last_stage: ray.actor.ActorHandle
    server_config: ServerConfig


@dataclass(frozen=True)
class StageResult:
    """Minimal stage result extended by profiling in M8."""

    rank: int
    max_memory_mib: float
    timelines: tuple[StepTimeline, ...]


@ray.remote(concurrency_groups={"compute": 1, "io": 8})
class StageActor:
    """Own one model partition and one rank in the torch process group."""

    def __init__(self, config: ExperimentConfig, rank: int) -> None:
        if config.server is None:
            raise ValueError("StageActor requires server config")
        self.config = config
        self.rank = rank
        self.world_size = config.server.world_size
        self.mailbox = Mailbox()
        self.client: ray.actor.ActorHandle | None = None
        self.peers: list[ray.actor.ActorHandle] = []
        self._attached = threading.Event()
        self._ready = threading.Event()
        self._step_specs: dict[int, Future[list[SequenceSpec]]] = {}
        self._step_lock = threading.Lock()
        self.device = torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.stage: USLPipelineStage | None = None
        self.schedule: ScheduleGPipe | Schedule1F1B | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.outbox: Outbox | None = None
        self.current_step = -1
        self.gradients: ClientGradients | None = None
        self.activation_offloader: AsyncActivationOffloader | None = None
        self.recorder = Recorder(config.profile.enabled)

    @ray.method(concurrency_group="io")
    def gpu_ids(self) -> list[str]:
        """Return Ray-assigned physical GPU identifiers before CUDA initialization."""
        return [str(item) for item in ray.get_runtime_context().get_accelerator_ids().get("GPU", [])]

    @ray.method(concurrency_group="io")
    def node_id(self) -> str:
        """Return this actor's Ray node identifier."""
        return ray.get_runtime_context().get_node_id()

    @ray.method(concurrency_group="io")
    def node_ip(self) -> str:
        """Return this actor's node address for torch distributed setup."""
        return ray.util.get_node_ip_address()

    @ray.method(concurrency_group="io")
    def distributed_endpoint(self) -> str:
        """Allocate the torch process-group rendezvous endpoint on rank zero."""
        if self.rank != 0:
            raise RuntimeError("Only rank zero allocates the process-group endpoint")
        return f"tcp://{self.node_ip()}:{find_free_port()}"

    @ray.method(concurrency_group="io")
    def clock(self) -> float:
        """Return a clock sample used to align profile spans."""
        return clock_ms()

    @ray.method(concurrency_group="io")
    def wait_ready(self, timeout_s: float) -> None:
        """Block client discovery until peer handles and setup are complete."""
        if not self._ready.wait(timeout=timeout_s):
            raise TimeoutError(f"Stage {self.rank} did not become ready within {timeout_s:g}s")

    @ray.method(concurrency_group="io")
    def push(self, key: MessageKey, blob: TensorBlob) -> None:
        """Receive a Ray-dereferenced boundary tensor."""
        self.mailbox.put(key, blob)

    @ray.method(concurrency_group="io")
    def parameter_state(self) -> dict[str, torch.Tensor]:
        """Return a CPU snapshot for numerical integration tests."""
        if self.model is None:
            raise RuntimeError("Stage model is not initialized")
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    @ray.method(concurrency_group="io")
    def abort(self, reason: str) -> None:
        """Unblock this actor after a peer failure."""
        self.mailbox.abort(reason)
        with self._step_lock:
            futures = list(self._step_specs.values())
        for future in futures:
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self._attached.set()

    @ray.method(concurrency_group="io")
    def set_peers(self, peers: list[ray.actor.ActorHandle]) -> None:
        """Install stage handles after all actors have been created."""
        self.peers = peers

    @ray.method(concurrency_group="io")
    def _set_client(self, client: ray.actor.ActorHandle) -> None:
        self.client = client
        self._attached.set()

    @ray.method(concurrency_group="io")
    def publish_profile(self, results: list[StageResult]) -> None:
        """Deliver aggregated stage results before either driver exits."""
        client = self._require_client()
        ray.get(client.set_server_results.remote(results))

    @ray.method(concurrency_group="io")
    def attach(self, client: ray.actor.ActorHandle, client_config: ExperimentConfig) -> ServerInfo:
        """Validate shared config and connect all stages to one client."""
        if self.rank != 0:
            raise RuntimeError("Client must attach to stage 0")
        differences = diff_shared_config(self.config, client_config)
        if differences:
            raise ValueError("Client/server config mismatch:\n" + "\n".join(differences))
        self.client = client
        self._attached.set()
        refs = [peer._set_client.remote(client) for index, peer in enumerate(self.peers) if index != 0]
        if refs:
            ray.get(refs)
        if self.config.server is None:
            raise RuntimeError("Missing server config")
        ranges = tuple(
            (item.start, item.stop)
            for item in SplitPlan(
                self.config.model.num_hidden_layers,
                self.config.split.split_point,
            ).server_stages(self.config.server.world_size)
        )
        return ServerInfo(self.world_size, ranges, self.node_id(), self.peers[-1], self.config.server)

    def _step_future(self, step: int) -> Future[list[SequenceSpec]]:
        with self._step_lock:
            return self._step_specs.setdefault(step, Future())

    @ray.method(concurrency_group="io")
    def _set_step(self, step: int, specs: list[SequenceSpec]) -> None:
        future = self._step_future(step)
        if not future.done():
            future.set_result(specs)

    @ray.method(concurrency_group="io")
    def begin_step(self, step: int, specs: list[SequenceSpec]) -> None:
        """Broadcast tensor-free sequence metadata and align the global step."""
        if self.rank != 0:
            raise RuntimeError("Client must call begin_step on stage 0")
        if step < 0 or step >= self.config.train.steps:
            raise ValueError(f"Step {step} is outside configured range")
        self._set_step(step, specs)
        refs = [peer._set_step.remote(step, specs) for index, peer in enumerate(self.peers) if index != 0]
        if refs:
            ray.get(refs)

    @ray.method(concurrency_group="compute")
    def setup(self, all_gpu_ids: list[str], init_method: str) -> None:
        """Initialize device, process group, partition, optimizer, and schedule."""
        del all_gpu_ids
        server = self.config.server
        if server is None:
            raise RuntimeError("Missing server config")
        if server.backend == "nccl":
            assigned = self.gpu_ids()
            if len(assigned) != 1:
                raise RuntimeError(f"Rank {self.rank} expected one GPU, got {assigned}")
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        dist.init_process_group(
            server.backend,
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device if server.backend == "nccl" else None,
        )
        dist.barrier(device_ids=[0] if server.backend == "nccl" else None)
        self._check_p2p()
        torch.manual_seed(self.config.run.seed)
        self.model = load_server_partition(self.config, self.rank).to(self.device).train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.learning_rate)
        self.outbox = Outbox(
            self.device,
            lambda key, start, end, nbytes: self.recorder.transfer("downlink", key, start, end, nbytes),
        )
        if server.activation_offload_mb and self.device.type == "cuda":
            self.activation_offloader = AsyncActivationOffloader(self.device)
        self.gradients = ClientGradients(self._receive_gradient) if self.rank == self.world_size - 1 else None
        dtype = next(self.model.parameters()).dtype
        shape = (
            self.config.batch.micro_batch_size,
            self.config.batch.max_seq_len,
            self.config.model.hidden_size,
        )
        input_meta = torch.empty(shape, dtype=dtype, device="meta")
        output_meta = torch.empty(shape, dtype=dtype, device="meta")
        self.stage = USLPipelineStage(
            self.model,
            self.rank,
            self.world_size,
            self.device,
            input_args=input_meta,
            output_args=output_meta,
            client_gradients=self.gradients,
            activation_offloader=self.activation_offloader,
            activation_offload_mb=server.activation_offload_mb,
            recorder=self.recorder,
        )
        loss_fn: Any = ClientLoss(self.gradients, self._send_activation) if self.gradients is not None else _unused_loss
        schedule_class = Schedule1F1B if self.config.pipeline.server_schedule == "1f1b" else ScheduleGPipe
        self.schedule = schedule_class(
            self.stage,
            n_microbatches=self.config.batch.num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
        self._ready.set()

    def _check_p2p(self) -> None:
        value = torch.tensor([1.0], device=self.device)
        dist.all_reduce(value)
        if value.item() != self.world_size:
            raise RuntimeError(f"Process-group self-check returned {value.item()}, expected {self.world_size}")

    def _require_client(self) -> ray.actor.ActorHandle:
        if self.client is None:
            raise RuntimeError("Client is not attached")
        return self.client

    def _receive_activation(self, microbatch: int) -> torch.Tensor:
        key = MessageKey(self.current_step, microbatch, "act_up")
        blob = self.mailbox.get(key, self.config.network.timeout_s)
        self.mailbox.discard(key)
        return blob.to_tensor(self.device, non_blocking=self.device.type == "cuda")

    def _receive_gradient(self, microbatch: int) -> torch.Tensor:
        key = MessageKey(self.current_step, microbatch, "grad_up")
        blob = self.mailbox.get(key, self.config.network.timeout_s)
        self.mailbox.discard(key)
        return blob.to_tensor(self.device, non_blocking=self.device.type == "cuda")

    def _send_activation(self, microbatch: int, tensor: torch.Tensor) -> None:
        if self.outbox is None:
            raise RuntimeError("Stage is not set up")
        self.outbox.submit(
            self._require_client(),
            MessageKey(self.current_step, microbatch, "act_down"),
            tensor,
        )

    def _send_gradient(self, microbatch: int, tensor: torch.Tensor) -> None:
        if self.outbox is None:
            raise RuntimeError("Stage is not set up")
        self.outbox.submit(
            self._require_client(),
            MessageKey(self.current_step, microbatch, "grad_down"),
            tensor,
        )

    @ray.method(concurrency_group="compute")
    def run(self) -> StageResult:
        """Run exactly the configured global steps and then release resources."""
        server = self.config.server
        if (
            server is None
            or self.stage is None
            or self.schedule is None
            or self.optimizer is None
            or self.outbox is None
        ):
            raise RuntimeError("StageActor.setup must complete before run")
        if not self._attached.wait(timeout=server.wait_client_s):
            raise TimeoutError(f"Stage {self.rank} timed out waiting for client attach")
        try:
            for step in range(self.config.train.steps):
                self.current_step = step
                self.stage.current_step = step
                specs = self._step_future(step).result(timeout=self.config.network.timeout_s)
                if len(specs) != self.config.batch.num_microbatches:
                    raise ValueError(
                        f"Step {step} expected {self.config.batch.num_microbatches} specs, got {len(specs)}"
                    )
                args: list[tuple[torch.Tensor, ...]]
                if self.rank == 0:
                    args = ClientInputs(len(specs), self._receive_activation, self._send_gradient)
                else:
                    args = [tuple() for _ in specs]
                self.stage.has_backward = True
                self.stage.clear_runtime_states()
                targets = [torch.tensor(index, device=self.device) for index in range(len(specs))]
                self.schedule._step_microbatches(args, microbatch_kwargs(specs), targets, None)
                with self.recorder.span(f"server/{self.rank}", "server_step", step, -1, self.device):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.outbox.flush()
                with self._step_lock:
                    self._step_specs.pop(step, None)
            max_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
            return StageResult(self.rank, max_memory, self.recorder.timelines())
        except BaseException as error:
            if self.client is not None:
                self.client.abort.remote(f"Server stage {self.rank} failed: {error}")
            raise
        finally:
            self.outbox.close()
            if self.activation_offloader is not None:
                self.activation_offloader.close()
            if dist.is_initialized():
                dist.destroy_process_group()


def _unused_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    del target
    return output.sum() * 0
