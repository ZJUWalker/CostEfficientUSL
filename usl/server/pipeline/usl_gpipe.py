import logging
from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)

import torch
import torch.distributed as dist
from torch.profiler import record_function

from torch.distributed.pipelining.schedules import _sorted_batch_p2p
from usl.server.pipeline.base_schedule import ServerPipelineScheduleSingle
from usl.offload import AsyncDoubleBufferGroupOffloadHandler, CpuOffloadHookWithOffloadHandler

if TYPE_CHECKING:
    from torch.distributed import Work

logger = logging.getLogger(__name__)


class ServerScheduleGPipe(ServerPipelineScheduleSingle):
    """
    The GPipe schedule for USL Server training.
    Will go through all the microbatches in a fill-drain manner.
    """

    def __init__(self, stage, n_microbatches, offload_activation_mb_num: int = 0):
        super().__init__(stage, n_microbatches)
        # ---- CUDA streams
        torch.cuda.set_stream(torch.cuda.Stream(self._stage.device))  # set cuda compute stream
        self.load_stream = torch.cuda.Stream(self._stage.device)  # set cuda load stream
        self.offload_stream = torch.cuda.Stream(self._stage.device)  # set cuda offload stream

        self.offload_activation_mb_num = offload_activation_mb_num
        if self.offload_activation_mb_num > 0:
            self.activation_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
                num_minibatch=self.offload_activation_mb_num,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
            )
            self.activation_offload_ctx = CpuOffloadHookWithOffloadHandler(self.activation_offload_handler)

    @property
    def offload_activation(self):
        return self.offload_activation_mb_num > 0

    def _step_microbatches(self):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """

        if not self._stage_initialized:
            self._initialize_stage()

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        if self.offload_activation:
            self.activation_offload_handler.start_fwd()  # mark the start of bwd
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()

                if i < self.offload_activation_mb_num:
                    with self.activation_offload_ctx:
                        _ = self._stage.forward_one_chunk(i)
                        # after ctx, the activation will be offloaded to CPU
                        self.activation_offload_handler.on_minibatch_commit_forward()
                else:
                    _ = self._stage.forward_one_chunk(i)
                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

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
        if self.offload_activation:
            self.activation_offload_handler.start_bwd()  # mark the start of bwd
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    work.wait()
                if i < self.offload_activation_mb_num:
                    self.activation_offload_handler.on_minibatch_commit_backward()
                self._stage.backward_one_chunk(i, last_backward=i == self._n_microbatches - 1)
                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Backwarded microbatch %s", self._stage.stage_index, i)

        # Return losses if there is a container passed in

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()

    def send_profile_res(self):
        profile_data = self._stage.profile_data
        res = {
            'profile': profile_data,
            'max_mem_alloc': round(self._stage.max_cuda_memory_allocated_during_fwd / 1024**2, 4),
            'server_fwd_time': 0,  # TODO add fwd time
            'server_fwd_send_time': 0,
            'server_bwd_time': 0,
            'server_bwd_send_time': 0,
            "server_offload_time_durations": self.activation_offload_handler.offload_time_durations if self.offload_activation else 0,
            "server_reload_time_durations": self.activation_offload_handler.reload_time_durations if self.offload_activation else 0,
            "file_suffix": f'soa_{self.offload_activation_mb_num}' if self.offload_activation else '',
        }
        self._stage.send_profile_res(res)
