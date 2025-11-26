import logging
from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)

import torch.distributed as dist
from torch.profiler import record_function

from torch.distributed.pipelining.schedules import _sorted_batch_p2p, _batch_p2p
from usl.server.pipeline.base_schedule import ServerPipelineScheduleSingle


if TYPE_CHECKING:
    from torch.distributed import Work

logger = logging.getLogger(__name__)


# TODO change to USL 1F1B
class ServerSchedule1F1B(ServerPipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(self):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """

        if not self._stage_initialized:
            self._initialize_stage()

        stage_warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index + 1,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work = None
        fwd_sends = []
        for _ in range(stage_warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if recv_work := _batch_p2p(fwd_recvs, desc="fwd_recv"):
                recv_work.wait()

            # Compute
            _ = self._stage.forward_one_chunk(fwd_mb_index)  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            if send_work:
                send_work.wait()

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != stage_warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last foward send is left for fuse with first 1B in 1B1F below
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
            # loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=None,
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
            _ = self._stage.forward_one_chunk(fwd_mb_index)  # type: ignore[index]

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
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=None,
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
