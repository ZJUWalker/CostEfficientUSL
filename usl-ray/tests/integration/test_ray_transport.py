"""Local Ray object-store transport integration test."""

from __future__ import annotations

import ray
import torch

from usl_ray.transport import Mailbox, MessageKey, Outbox, TensorBlob


@ray.remote
class Receiver:
    """Small actor exercising automatic top-level ObjectRef resolution."""

    def __init__(self) -> None:
        self.mailbox = Mailbox()

    def push(self, key: MessageKey, blob: TensorBlob) -> None:
        """Receive a dereferenced blob on this actor's node."""
        assert isinstance(blob, TensorBlob)
        self.mailbox.put(key, blob)

    def value(self, key: MessageKey) -> torch.Tensor:
        """Return one received tensor."""
        return self.mailbox.get(key, 5.0).to_tensor()


def test_outbox_uses_ray_object_refs() -> None:
    """Put-and-push transfers a tensor without sockets or polling loops."""
    ray.init(address="local", num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        receiver = Receiver.remote()
        outbox = Outbox("cpu")
        key = MessageKey(0, 0, "act_up")
        source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        outbox.submit(receiver, key, source)
        outbox.flush()
        actual = ray.get(receiver.value.remote(key))
        torch.testing.assert_close(actual, source)
        outbox.close()
    finally:
        ray.shutdown()
