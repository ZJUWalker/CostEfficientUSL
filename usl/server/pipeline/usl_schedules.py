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
    "Schedule1F1B",
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


class ServerScheduleGPipe(ServerPipelineScheduleSingle):
    """
    The GPipe schedule for USL Server training.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        print(f"Rank {dist.get_rank()}: arg_mbs: {arg_mbs}, kwarg_mbs: {kwarg_mbs}")

        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()

                _ = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

            # self._maybe_compute_loss(self._stage, output, target_mbs, i)# Server don't need to compute loss in USL training

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    work.wait()

                # loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(i, loss=None, last_backward=i == self._n_microbatches - 1)
                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Backwarded microbatch %s", self._stage.stage_index, i)

        # Return losses if there is a container passed in
        # self._update_losses(self._stage, losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()


# TODO change to USL 1F1B
class Schedule1F1B(ServerPipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work = None
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if recv_work := _batch_p2p(fwd_recvs, desc="fwd_recv"):
                recv_work.wait()

            # Compute
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            if send_work:
                send_work.wait()

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last foward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            if fuse_work := _batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"):
                fuse_work.wait()

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            if fuse_work := _batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"):
                fuse_work.wait()

            # Now do the fwd
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")

        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            if recv_work := _batch_p2p(bwd_recvs, desc="bwd_recv"):
                recv_work.wait()

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Clear previous chunk's backward sends (hopefully they have well finished)
            if send_work:
                send_work.wait()

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        if send_work:
            send_work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)


def get_schedule_class(schedule_name: str):
    """
    Maps a schedule name (case insensitive) to its corresponding class object.

    Args:
        schedule_name (str): The name of the schedule.
    """
    schedule_map = {
        "1F1B": Schedule1F1B,
        "GPipe": ServerScheduleGPipe,
        "PipelineScheduleSingle": ServerPipelineScheduleSingle,
    }
    lowercase_keys = {k.lower(): k for k in schedule_map.keys()}
    lowercase_schedule_name = schedule_name.lower()
    if lowercase_schedule_name not in lowercase_keys:
        raise ValueError(f"Unknown schedule name '{schedule_name}'. The valid options are {list(schedule_map.keys())}")
    return schedule_map[lowercase_keys[lowercase_schedule_name]]
