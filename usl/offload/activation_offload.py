# This file is largely inspired by and partially follows the structure of
# ``transformer_engine.pytorch.cpu_offload`` in
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/cpu_offload.py
"""Functionality for CPU offloading of activation tensors saved for backward pass."""
import math
import time
import warnings
from typing import Any, List, Tuple

import torch


def tensor_need_offloading_checker(tensor: torch.Tensor):
    if not tensor.is_cuda:
        return False
    # This is a bit tricky; if a PyTorch tensor is a view tensor, then its _base attribute
    # will be the previous tensor. The judgement here will filter out tensors that are not
    # activations, such as the transposed weights (weight.T) of the linear layer.
    if tensor._base is not None:
        return not tensor._base.is_leaf
    return not tensor.is_leaf or not tensor.requires_grad


class TensorState:

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor: torch.Tensor = tensor
        self.tensor_meta = [(tensor.size(), tensor.dtype)]
        self.ref_cnt = 1
        self.device = tensor.device
        self.offloaded = False
        self.reloaded = False
        self.prefetch_buffer = None

    def add_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.data_ptr() == self.tensor.data_ptr()
        assert tensor.dtype == self.tensor.dtype
        self.ref_cnt += 1
        self.tensor_meta.append((tensor.size(), tensor.dtype))

    def get_ref_cnt(self) -> int:
        return self.ref_cnt

    def get_tensor(self) -> torch.Tensor:
        return self.tensor

    def get_reloaded_tensor(self, ref_cnt) -> torch.Tensor:
        self.ref_cnt -= 1
        assert self.ref_cnt >= 0
        assert ref_cnt >= 1 and ref_cnt <= len(self.tensor_meta)
        assert self.prefetch_buffer is not None and self.reloaded
        return self.prefetch_buffer.view(self.tensor_meta[ref_cnt - 1][0])

    def offload(self, pin_memory=True) -> None:
        assert not self.offloaded
        self.cpu_backup = torch.empty(self.tensor.size(), dtype=self.tensor.dtype, layout=self.tensor.layout, device="cpu", pin_memory=pin_memory)
        self.cpu_backup.copy_(self.tensor, non_blocking=pin_memory)
        self.tensor = None
        self.offloaded = True

    def create_prefetch_buffer(self) -> None:
        self.prefetch_buffer = torch.empty(self.cpu_backup.size(), dtype=self.cpu_backup.dtype, layout=self.cpu_backup.layout, device=self.device)

    def reload(self) -> None:
        assert not self.reloaded and self.offloaded
        self.prefetch_buffer.copy_(self.cpu_backup, non_blocking=True)
        self.reloaded = True


class CpuOffloadSavedTensorHook:

    def __init__(self) -> None:
        self.inside_context = False

    def __enter__(self):
        if not torch.is_grad_enabled():
            return

        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(self.on_save_for_backward, self.on_get_saved_tensor)

    def __exit__(self, *args: Any):
        if not torch.is_grad_enabled():
            return

        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """On save for backward."""
        raise NotImplementedError(
            "`on_save_for_backward: Callable[[torch.Tensor], Any]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks"
        )

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """On get saved tensor."""
        raise NotImplementedError(
            "`on_get_saved_tensors: Callable[[Any], torch.Tensor]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks"
        )


class OffloadHandler:
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. " "Inherit this class and implement your custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        """Tensor pop."""
        raise NotImplementedError("`tensor_pop is not implented in OffloadHandler class. " "Inherit this class and implement your custom tensor_pop.")


class CpuOffloadHookWithOffloadHandler(CpuOffloadSavedTensorHook):
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(self, offload_handler: OffloadHandler, handler_extra_kwargs={}, debug=False) -> None:  # pylint: disable=dangerous-default-value
        self.debug = debug
        self.offload_handler = offload_handler
        self.handler_extra_kwargs = handler_extra_kwargs
        super().__init__()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        return tensor


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(self, num_minibatch: int = 1, tensor_need_offloading_checker=tensor_need_offloading_checker, debug=False) -> None:
        super().__init__()

        self.num_minibatch = num_minibatch
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug
        self.minibatch_idx_reset()

    def minibatch_idx_reset(self):
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_mb_idx, self.tensor_count_current_mb_idx = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def start_fwd(self):
        self.minibatch_idx_reset()

    def start_bwd(self):
        self.current_mb_idx = -1

    def on_minibatch_commit_forward(self):
        """On minibatch commit forward."""
        # finishing up with updating current minibatch and tensor count
        self.current_mb_idx += 1  # increment
        self.tensor_count_current_mb_idx = 0  # reset

    def on_minibatch_commit_backward(self):
        """On minibatch commit backward."""
        # finishing up with updating current minibatch and tensor count
        self.current_mb_idx += 1
        assert self.current_mb_idx >= 0

    @staticmethod
    def offload(src_tensor: torch.Tensor, pin_memory=True):
        """Offload."""

        cpu_backup = torch.empty(src_tensor.size(), dtype=src_tensor.dtype, layout=src_tensor.layout, device="cpu", pin_memory=pin_memory)

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state: Tuple[torch.device, torch.Tensor], non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    # 'pack'
    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_mb_idx, self.tensor_count_current_mb_idx)
        self.tensor_count_current_mb_idx += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if self.current_mb_idx < self.num_minibatch and self.tensor_need_offloading_checker(tensor):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after minibatch commit
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    # 'unpack'
    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    def __init__(
        self,
        num_minibatch,  # must be <= actual number of groups (number of commits)
        num_minibatch_prefetch=2,  # just set to 1 for now
        load_stream: torch.cuda.Stream = None,
        offload_stream: torch.cuda.Stream = None,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
        debug=False,
    ) -> None:
        super().__init__(
            num_minibatch=num_minibatch,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
            debug=debug,
        )
        self.num_prefetch_mb = num_minibatch_prefetch
        self.offloaded_tensor_buffers = [[] for _ in range(num_minibatch)]

        # allocate streams and events for synchronization
        self.d2h_stream = offload_stream if offload_stream is not None else torch.cuda.Stream()
        self.h2d_stream = load_stream if load_stream is not None else torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
        self.offload_start_timestamp: List[float] = []
        self.offload_time_durations: List[float] = []
        self.reload_start_timestamp: List[float] = []
        self.reload_time_durations: List[float] = []
        self.h2d_start_events: List[torch.cuda.Event] = []
        self.h2d_finish_events: List[torch.cuda.Event] = []
        self.d2h_start_events: List[torch.cuda.Event] = []
        self.d2h_finish_events: List[torch.cuda.Event] = []
        self.compute_stream_bwd_start_events: List[torch.cuda.Event] = []
        for _ in range(self.num_minibatch):
            self.offload_time_durations.append(0)
            self.offload_start_timestamp.append(None)
            self.reload_time_durations.append(0)
            self.reload_start_timestamp.append(None)
            self.h2d_start_events.append(torch.cuda.Event(enable_timing=True))
            self.h2d_finish_events.append(torch.cuda.Event(enable_timing=True))
            self.d2h_start_events.append(torch.cuda.Event(enable_timing=True))
            self.d2h_finish_events.append(torch.cuda.Event(enable_timing=True))
            self.compute_stream_bwd_start_events.append(torch.cuda.Event(enable_timing=True))

        self.total_offload_size = 0

    def start_bwd(self):
        self.next_mb_to_fetch = 0
        return super().start_bwd()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        if self.tensor_need_offloading_checker(tensor):
            # obtain a unique tensor tag
            if self.current_mb_idx < self.num_minibatch:
                # We use the data_ptr as the tensor tag to eliminate some views of the same tensor.
                # It's worth noting that we preserve tensors, so the data pointers of different tensors
                # within the same minibatch will not be identical.
                tensor_tag = (self.current_mb_idx, tensor.data_ptr())
                if tensor_tag not in self.tensor_tag_to_state:
                    self.tensor_tag_to_state[tensor_tag] = TensorState(tensor)
                else:
                    self.tensor_tag_to_state[tensor_tag].add_tensor(tensor)
                tensor_tag = (tensor_tag, self.tensor_tag_to_state[tensor_tag].get_ref_cnt())
            else:
                tensor_tag = (self.current_mb_idx, self.tensor_count_current_mb_idx)
                self.tensor_count_current_mb_idx += 1
                assert tensor_tag not in self.tensor_tag_to_state
                self.tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        if isinstance(tensor_tag[0], tuple):
            tensor_tag, ref_cnt = tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor_or_state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(tensor_or_state, TensorState):
            tensor = tensor_or_state.get_reloaded_tensor(ref_cnt)
            if tensor_or_state.get_ref_cnt() > 0:
                self.tensor_tag_to_state[tensor_tag] = tensor_or_state
        else:
            tensor = tensor_or_state
        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, TensorState)
        return tensor

    def bulk_offload_group(self, mb_to_offload: int):
        # the copying of this minibatch should wait for the computation stream
        self.d2h_stream.wait_stream(self.compute_stream)
        """Bulk offload minibatch."""
        # start_offload=time.time()
        self.offload_start_timestamp[mb_to_offload] = time.time()
        self.d2h_stream.record_event(self.d2h_start_events[mb_to_offload])
        # print('record d2h start event for mb_idx', mb_to_offload)
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                mb_idx, _ = tensor_tag
                if mb_idx == mb_to_offload and isinstance(state, TensorState):
                    tensor_on_device = state.get_tensor()
                    assert self.tensor_need_offloading_checker(tensor_on_device)
                    # if offload, return the reference to cpu copy
                    if tensor_on_device is not None:
                        self.total_offload_size += tensor_on_device.numel() * tensor_on_device.element_size()
                        state.offload()
                        # save the tensor since this the copy of this tensor has not yet finished
                        self.offloaded_tensor_buffers[self.current_mb_idx].append(tensor_on_device)
        # print(f"offloading tensor size: {self.total_offload_size / (1024.0**3):.5f} GiB")

    def synchronize_on_group_commit_forward(self, current_mb_idx: int):
        """Synchronize on minibatch commit forward."""
        if current_mb_idx < self.num_minibatch:
            # perform bulk offloading
            self.bulk_offload_group(current_mb_idx)
            self.d2h_stream.record_event(self.d2h_finish_events[current_mb_idx])
            # print('record d2h finish event for mb_idx', current_mb_idx)
        # wait for the previous minibatch to finish offloading,we need to clear the GPU buffer of the previous minibatch
        if current_mb_idx > 0 and current_mb_idx < self.num_minibatch:
            pre_mb_idx = current_mb_idx - 1
            self.compute_stream.wait_event(self.d2h_finish_events[pre_mb_idx])
            self.d2h_finish_events[pre_mb_idx].synchronize()
            # print(f'elapsed time for offloading mb_idx {pre_mb_idx}: {self.offload_time_durations[pre_mb_idx]}')
            self.offload_time_durations[pre_mb_idx] = self.d2h_start_events[pre_mb_idx].elapsed_time(self.d2h_finish_events[pre_mb_idx])
            # release tensors since the offloading has finished
            self.offloaded_tensor_buffers[pre_mb_idx].clear()

    def on_minibatch_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_mb_idx)
        super().on_minibatch_commit_forward()

    def bulk_reload_group(self, mb_to_reload: int):
        """Bulk reload minibatch."""
        assert mb_to_reload < self.num_minibatch

        # allocating tensors in the current stream allows subsequent ops in current streams
        # to reuse the GPU memory.
        for tensor_label, state in self.tensor_tag_to_state.items():
            mb_idx, _ = tensor_label
            if mb_idx == mb_to_reload:
                if isinstance(state, TensorState):
                    state.create_prefetch_buffer()
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self.tensor_tag_to_state.items():
                mb_idx, _ = tensor_label
                if mb_idx == mb_to_reload:
                    if isinstance(state, TensorState):
                        state.reload()

    def on_minibatch_commit_backward(self):
        # first we need to wait for the last minibatch activation to finish offloading
        if self.current_mb_idx == -1:
            self.h2d_stream.wait_event(self.d2h_finish_events[self.num_minibatch - 1])
            self.offloaded_tensor_buffers[self.num_minibatch - 1].clear()
        # start reloading the activations
        self.current_mb_idx += 1
        assert self.current_mb_idx >= 0 and self.current_mb_idx < self.num_minibatch
        if self.current_mb_idx == self.num_minibatch - 1:
            self.total_offload_size = 0

        for mb_idx in range(self.num_minibatch):
            assert len(self.offloaded_tensor_buffers[mb_idx]) == 0, (
                "num_offload_sync_layers * num_offload_layers + 1 cannot be greater than" " the number of all layers."
            )

        # decide the range of minibatches to prefetch
        should_prefetch_until_mb_idx = (
            self.current_mb_idx + self.num_prefetch_mb
        )  # always set to 2 for memory saving,and it's enough to be overlapped for now.
        # num_prefetch_mb: the number of minibatches to prefetch
        should_prefetch_until_mb_idx = min(should_prefetch_until_mb_idx, self.num_minibatch)
        # print(f"prefetching  from mb_idx {self.next_mb_to_fetch} until mb_idx {should_prefetch_until_mb_idx}")
        # do prefetch minibatch
        for mb_idx_to_prefetch in range(self.next_mb_to_fetch, should_prefetch_until_mb_idx):
            # record the event in the compute stream, for h2d to wait ( wait for pre minibatch bwd to finish)
            self.compute_stream.record_event(self.compute_stream_bwd_start_events[mb_idx_to_prefetch])

            # start of h2d should wait for the compute and the d2h
            self.h2d_stream.wait_event(self.compute_stream_bwd_start_events[mb_idx_to_prefetch])

            self.reload_start_timestamp[mb_idx_to_prefetch] = time.time()
            self.h2d_stream.record_event(self.h2d_start_events[mb_idx_to_prefetch])

            # recover tensors (copy back from host)
            self.bulk_reload_group(mb_idx_to_prefetch)

            # record an event for the backward of this layer to wait
            self.h2d_stream.record_event(self.h2d_finish_events[mb_idx_to_prefetch])

        # update the next mb to prefetch
        self.next_mb_to_fetch = min(self.num_minibatch, should_prefetch_until_mb_idx)

        # torch.cuda.synchronize(self.h2d_stream)
        # wait for the current minibatch to finish loading
        if self.current_mb_idx < self.num_minibatch:
            self.compute_stream.wait_event(self.h2d_finish_events[self.current_mb_idx])
            self.h2d_finish_events[self.current_mb_idx].synchronize()
            self.reload_time_durations[self.current_mb_idx] = self.h2d_start_events[self.current_mb_idx].elapsed_time(
                self.h2d_finish_events[self.current_mb_idx]
            )
            # print(f'elapsed time for reloading mb_idx {self.current_mb_idx}: {self.offload_time_durations[self.current_mb_idx]}')

        # if self.current_mb_idx == self.num_minibatch - 1:
        #     last_idx = self.num_minibatch - 1
        #     torch.cuda.synchronize(self.h2d_stream)  # 等待所有 reload 事件完成
        #     self.reload_time_durations[last_idx] = \
        #         self.h2d_start_events[last_idx].elapsed_time(self.h2d_finish_events[last_idx])
        #     print(f"[Finalize] elapsed time for reloading mb_idx {last_idx}: "
        #         f"{self.reload_time_durations[last_idx]:.3f} ms")
