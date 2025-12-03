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
from typing_extensions import deprecated

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
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = lambda: torch.tensor(0.0)  # dummy loss function TODO delete anything about loss and loss_fn
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None
        # print(f"Rank {dist.get_rank()}: _has_backward: {self._has_backward}")
        logger.info("Using %s", self.__class__.__name__)

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
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward
        self._stage_initialized = False

    def _initialize_stage(self):
        self._stage._prepare_forward_infra(self._n_microbatches)
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
        # Run microbatches
        self._step_microbatches()

    # TODO do profile here
    def send_profile_res(self):
        profile_data = self._stage.profile_data
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

        res = {
            'profile': profile_data,
            'max_mem_alloc': round(self._stage.max_cuda_memory_allocated / 1024**2, 4),
            'server_fwd_time': 0,
            'server_fwd_send_time': 0,
            'server_bwd_time': 0,
            'server_bwd_send_time': 0,
            "server_offload_time_durations": 0,
            "server_reload_time_durations": 0,
            "file_suffix": '',
        }
        self._stage.send_profile_res(res)
