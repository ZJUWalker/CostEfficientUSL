# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.distributed as dist
from torch.profiler import record_function

from torch.distributed.pipelining.microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from torch.distributed.pipelining.schedules import _sorted_batch_p2p, _batch_p2p
from usl.server.pipeline.server_stage import _ServerPipelineStageBase


if TYPE_CHECKING:
    from torch.distributed import Work

__all__ = [
    "get_schedule_class",
    "ServerPipelineScheduleSingle",
    "ServerSchedule1F1B",
    "ServerScheduleGPipe",
]

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    UNSHARD = 4
    RESHARD = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10

    def __str__(self):
        str_map = {
            _ComputationType.FORWARD: "F",
            _ComputationType.BACKWARD_INPUT: "I",
            _ComputationType.BACKWARD_WEIGHT: "W",
            _ComputationType.UNSHARD: "UNSHARD",
            _ComputationType.RESHARD: "RESHARD",
            _ComputationType.SEND_F: "SEND_F",
            _ComputationType.RECV_F: "RECV_F",
            _ComputationType.SEND_B: "SEND_B",
            _ComputationType.RECV_B: "RECV_B",
            _ComputationType.FULL_BACKWARD: "B",
        }
        return str_map[self]

    @staticmethod
    def from_str(action):
        if action == "F":
            return _ComputationType.FORWARD
        elif action == "I":
            return _ComputationType.BACKWARD_INPUT
        elif action == "W":
            return _ComputationType.BACKWARD_WEIGHT
        elif action == "UNSHARD":
            return _ComputationType.UNSHARD
        elif action == "RESHARD":
            return _ComputationType.RESHARD
        elif action == "SEND_F":
            return _ComputationType.SEND_F
        elif action == "RECV_F":
            return _ComputationType.RECV_F
        elif action == "SEND_B":
            return _ComputationType.SEND_B
        elif action == "RECV_B":
            return _ComputationType.RECV_B
        elif action == "B":
            return _ComputationType.FULL_BACKWARD
        else:
            raise RuntimeError(f"Invalid computation type {action}")


FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD


class _ServerPipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = lambda: torch.tensor(0.0)  # dummy loss function TODO delete anything about loss and loss_fn
        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None
        print(f"Rank {dist.get_rank()}: _has_backward: {self._has_backward}")

        # Holds the losses for each microbatch.
        self._internal_losses: List[torch.Tensor] = []  # TODO delete this
        logger.info("Using %s", self.__class__.__name__)

    """
    no need for compute ,save ,retrive losses!!!!!!!!!!
    """
    # def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
    #     if stage.is_last and self._has_backward:
    #         loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
    #         self._internal_losses.append(loss)

    # def _maybe_get_loss(self, stage, mb_index):
    #     valid_index = 0 <= mb_index < len(self._internal_losses)
    #     if stage.is_last and self._has_backward and valid_index:
    #         return self._internal_losses[mb_index]
    #     elif len(self._internal_losses) != 0 and not valid_index:
    #         raise RuntimeError(f"Loss for microbatch {mb_index} is not available. " f"Available losses for microbatches: {self._internal_losses}")
    #     else:
    #         return None

    # def _update_losses(self, stages, losses):
    #     """
    #     Update the losses to those in the internal state
    #     """
    #     # if stages not a list turn into a list
    #     if not isinstance(stages, list):
    #         stages = [stages]
    #     contains_last_stage = any(stage.is_last for stage in stages)

    #     # Return losses if there is a container passed in
    #     if contains_last_stage and losses is not None:
    #         if len(self._internal_losses) != self._n_microbatches:
    #             raise RuntimeError(f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}")

    #         # Clean external container first
    #         losses.clear()
    #         # Copy internal losses to external container
    #         losses.extend(self._internal_losses)

    #     self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        raise NotImplementedError

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(f"Expecting {self._n_microbatches} {name} but got {len(mbs)}")

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    # def _compute_loss(self, output, target):
    #     return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


class ServerPipelineScheduleSingle(_ServerPipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: _ServerPipelineStageBase,
        n_microbatches: int,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward
        self._stage_initialized = False

    def _initialize_stage(self, args, kwargs):
        self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs)
        if self._has_backward:
            self._stage._prepare_backward_infra(self._n_microbatches)
        self._stage_initialized = True

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """

        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        # if self._stage.is_last:
        #     return self._merge_outputs(self._stage.output_chunks)
        # else:
        return None

    # TODO do profile here
    def send_profile_res(self):
        self._stage.send_profile_res()
        #         {
        #     "profile": None,
        #     "max_mem_alloc": round(self.max_cuda_memory_allocated / 1024**2, 4),
        #     "server_fwd_time": self.server_fwd_time,
        #     "server_fwd_send_time": self.server_fwd_send_time,
        #     "server_bwd_time": self.server_bwd_time,
        #     "server_bwd_send_time": self.server_bwd_send_time,
        #     "server_offload_time_durations": (
        #         self.activation_offload_handler.offload_time_durations if self.server_args.offload_activation else 0
        #     ),
        #     "server_reload_time_durations": (
        #         self.activation_offload_handler.reload_time_durations if self.server_args.offload_activation else 0
        #     ),
        #     "file_suffix": f'soa_{self.server_args.offload_activation_mb_num}' if self.server_args.offload_activation else '',
        # }
