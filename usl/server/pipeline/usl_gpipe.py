import logging
from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)

import torch.distributed as dist
from torch.profiler import record_function

from torch.distributed.pipelining.schedules import _sorted_batch_p2p
from usl.server.pipeline.base_schedule import ServerPipelineScheduleSingle


if TYPE_CHECKING:
    from torch.distributed import Work

logger = logging.getLogger(__name__)


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
        # print(f"Rank {dist.get_rank()}: arg_mbs: {arg_mbs}, kwarg_mbs: {kwarg_mbs}")

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
