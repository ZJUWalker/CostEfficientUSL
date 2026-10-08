"""Adapters between Ray boundary messages and stock torch pipeline schedules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining import PipelineStage

from usl_ray.offload import AsyncActivationOffloader
from usl_ray.profile.recorder import Recorder

ReceiveTensor = Callable[[int], torch.Tensor]
SendTensor = Callable[[int, torch.Tensor], None]


class ClientInputs(list[tuple[torch.Tensor]]):
    """Lazy first-stage inputs backed by the client activation mailbox."""

    def __init__(self, count: int, receiver: ReceiveTensor, gradient_sender: SendTensor) -> None:
        super().__init__([(torch.empty(0),)] * count)
        self._receiver = receiver
        self._gradient_sender = gradient_sender
        self._cache: dict[int, tuple[torch.Tensor]] = {}

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor] | list[tuple[torch.Tensor]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index not in self._cache:
            activation = self._receiver(index).detach().requires_grad_(True)
            activation.register_hook(lambda gradient, mb=index: self._gradient_sender(mb, gradient.detach()))
            self._cache[index] = (activation,)
        return self._cache[index]


class ClientGradients:
    """Preload last-stage output gradients outside the measured backward span."""

    def __init__(self, receiver: ReceiveTensor) -> None:
        self._receiver = receiver
        self._cache: dict[int, torch.Tensor] = {}

    def prepare(self, index: int) -> None:
        """Receive a gradient once."""
        if index not in self._cache:
            self._cache[index] = self._receiver(index)

    def pop(self, index: int) -> torch.Tensor:
        """Return a previously received gradient."""
        try:
            return self._cache.pop(index)
        except KeyError as error:
            raise RuntimeError(f"Client gradient {index} was not prepared") from error


class _ClientLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        output: torch.Tensor,
        index: int,
        gradients: ClientGradients,
        activation_sender: SendTensor,
    ) -> torch.Tensor:
        ctx.index = index
        ctx.gradients = gradients
        activation_sender(index, output.detach())
        return output.new_zeros(())

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        del gradient
        return ctx.gradients.pop(ctx.index), None, None, None


class ClientLoss:
    """Stock pipeline ``loss_fn`` that bridges outputs and client gradients."""

    def __init__(self, gradients: ClientGradients, activation_sender: SendTensor) -> None:
        self.gradients = gradients
        self.activation_sender = activation_sender

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Send one activation and return a differentiable scalar placeholder."""
        index = int(target.item())
        return _ClientLoss.apply(output, index, self.gradients, self.activation_sender)


class USLPipelineStage(PipelineStage):
    """PipelineStage that prepares the client gradient before timed backward."""

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        *,
        input_args: torch.Tensor,
        output_args: torch.Tensor,
        client_gradients: ClientGradients | None = None,
        activation_offloader: AsyncActivationOffloader | None = None,
        activation_offload_mb: int = 0,
        recorder: Recorder | None = None,
    ) -> None:
        super().__init__(
            submodule,
            stage_index,
            num_stages,
            device,
            input_args=input_args,
            output_args=output_args,
        )
        self.client_gradients = client_gradients
        self.activation_offloader = activation_offloader
        self.activation_offload_mb = activation_offload_mb
        self.recorder = recorder
        self.current_step = -1

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Offload selected stage activations while retaining stock communication."""
        context = (
            self.activation_offloader.context(offload_activations=True)
            if self.activation_offloader is not None and fwd_chunk_id < self.activation_offload_mb
            else nullcontext()
        )
        if self.recorder is None:
            with context:
                return super().forward_one_chunk(fwd_chunk_id, args, kwargs)
        with self.recorder.span(
            f"server/{self.stage_index}",
            "server_forward",
            self.current_step,
            fwd_chunk_id,
            self.device,
        ):
            with context:
                return super().forward_one_chunk(fwd_chunk_id, args, kwargs)

    def _get_init_p2p_neighbors_ops(self) -> list[dist.P2POp]:
        """Initialize torch P2P with matching send/receive dtypes.

        Torch 2.9.1 creates an int64 send tensor and float32 receive tensor in
        the stock implementation. Gloo aborts the process on the resulting
        8-byte versus 4-byte transfer. NCCL also requires matching metadata.
        """
        operations: list[dist.P2POp] = []
        next_rank = self.stage_index_to_group_rank.get(self.stage_index + 1)
        previous_rank = self.stage_index_to_group_rank.get(self.stage_index - 1)
        receive = torch.zeros(1, dtype=torch.float32, device=self.device)
        send = torch.tensor([float(self.stage_index)], dtype=torch.float32, device=self.device)
        if not self.is_first:
            operations.append(dist.P2POp(dist.irecv, receive, group_peer=previous_rank, group=self.group))
        if not self.is_last:
            operations.append(dist.P2POp(dist.isend, send, group_peer=next_rank, group=self.group))
        if not self.is_first:
            operations.append(dist.P2POp(dist.isend, send, group_peer=previous_rank, group=self.group))
        if not self.is_last:
            operations.append(dist.P2POp(dist.irecv, receive, group_peer=next_rank, group=self.group))
        return operations

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss: torch.Tensor | None = None,
        full_backward: bool = True,
        last_backward: bool = False,
    ) -> None:
        """Fetch last-stage client gradients before entering torch backward."""
        if self.is_last:
            if self.client_gradients is None:
                raise RuntimeError("Last stage requires a client gradient source")
            self.client_gradients.prepare(bwd_chunk_id)
        if self.recorder is None:
            super().backward_one_chunk(bwd_chunk_id, loss, full_backward, last_backward)
            return
        with self.recorder.span(
            f"server/{self.stage_index}",
            "server_backward",
            self.current_step,
            bwd_chunk_id,
            self.device,
        ):
            super().backward_one_chunk(bwd_chunk_id, loss, full_backward, last_backward)


def microbatch_kwargs(sequence_specs: Sequence[Any]) -> list[dict[str, Any]]:
    """Build stock-schedule kwargs for every server rank."""
    return [{"sequence": item} for item in sequence_specs]
