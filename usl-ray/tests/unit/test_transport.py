"""Transport primitive tests."""

from __future__ import annotations

import threading

import pytest
import torch

from usl_ray.transport import Mailbox, MessageKey, TensorBlob


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_tensor_blob_round_trip(dtype: torch.dtype) -> None:
    """Byte storage preserves dtypes NumPy cannot represent, including bfloat16."""
    source = torch.arange(12).reshape(3, 4).to(dtype)
    blob = TensorBlob.from_tensor(source)
    actual = blob.to_tensor()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, source)
    assert blob.nbytes == source.numel() * source.element_size()


def test_mailbox_accepts_arrival_before_waiter() -> None:
    """A producer can resolve a key before a consumer requests it."""
    mailbox = Mailbox()
    key = MessageKey(1, 2, "act_up")
    blob = TensorBlob.from_tensor(torch.ones(2))
    mailbox.put(key, blob)
    assert mailbox.get(key, 0.1) is blob


def test_mailbox_accepts_waiter_before_arrival() -> None:
    """A future created by the consumer is resolved by a later producer."""
    mailbox = Mailbox()
    key = MessageKey(2, 3, "grad_down")
    blob = TensorBlob.from_tensor(torch.ones(2))
    thread = threading.Thread(target=mailbox.put, args=(key, blob))
    thread.start()
    assert mailbox.get(key, 1.0) is blob
    thread.join()


def test_mailbox_timeout_names_key() -> None:
    """Timeout diagnostics identify the missing boundary message."""
    key = MessageKey(4, 5, "act_down")
    with pytest.raises(TimeoutError, match="step=4.*microbatch=5"):
        Mailbox().get(key, 0.001)


def test_mailbox_abort_fails_future_and_new_waiters() -> None:
    """Abort unblocks existing and future consumers."""
    mailbox = Mailbox()
    first = mailbox.future(MessageKey(1, 0, "act_up"))
    mailbox.abort("peer failed")
    with pytest.raises(RuntimeError, match="peer failed"):
        first.result()
    with pytest.raises(RuntimeError, match="peer failed"):
        mailbox.get(MessageKey(1, 1, "act_up"), 0.1)
