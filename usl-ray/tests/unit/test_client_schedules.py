"""Client schedule ordering tests without Ray or CUDA."""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from usl_ray.client.schedules import SCHEDULES


def _ready(value: str) -> Future[str]:
    future: Future[str] = Future()
    future.set_result(value)
    return future


class FakeTrainer:
    """Record all schedule calls."""

    server_world_size = 2

    def __init__(self) -> None:
        self.calls: list[str] = []

    def head_forward(self, microbatch: int) -> None:
        self.calls.append(f"hf{microbatch}")

    def activation(self, microbatch: int) -> Future[str]:
        self.calls.append(f"ra{microbatch}")
        return _ready(f"a{microbatch}")

    def tail_forward(self, microbatch: int, activation: object) -> None:
        assert activation == f"a{microbatch}"
        self.calls.append(f"tf{microbatch}")

    def tail_backward(self, microbatch: int) -> float:
        self.calls.append(f"tb{microbatch}")
        return 1.0

    def gradient(self, microbatch: int) -> Future[str]:
        self.calls.append(f"rg{microbatch}")
        return _ready(f"g{microbatch}")

    def head_backward(self, microbatch: int, gradient: object) -> None:
        assert gradient == f"g{microbatch}"
        self.calls.append(f"hb{microbatch}")

    def swap_to_tail(self) -> None:
        self.calls.append("swap_tail")

    def swap_to_head(self) -> None:
        self.calls.append("swap_head")

    def step_tail(self) -> None:
        self.calls.append("step_tail")

    def step_head(self) -> None:
        self.calls.append("step_head")


def test_splitmind_has_readable_fill_drain_order() -> None:
    """The paper schedule has no polling or hidden state machine."""
    trainer = FakeTrainer()
    loss = SCHEDULES["splitmind"](trainer, [0, 1])
    assert loss == 2.0
    assert trainer.calls == [
        "hf0",
        "hf1",
        "swap_tail",
        "ra0",
        "tf0",
        "tb0",
        "ra1",
        "tf1",
        "tb1",
        "step_tail",
        "swap_head",
        "rg0",
        "hb0",
        "rg1",
        "hb1",
        "step_head",
    ]


@pytest.mark.parametrize("mode", sorted(SCHEDULES))
def test_every_schedule_consumes_each_microbatch(mode: str) -> None:
    """Every supported schedule performs one forward and backward per partition."""
    trainer = FakeTrainer()
    loss = SCHEDULES[mode](trainer, [0] if mode == "sequential" else [0, 1, 2, 3])
    expected = 1 if mode == "sequential" else 4
    assert loss == float(expected)
    assert sum(call.startswith("hf") for call in trainer.calls) == expected
    assert sum(call.startswith("tf") for call in trainer.calls) == expected
    assert sum(call.startswith("tb") for call in trainer.calls) == expected
    assert sum(call.startswith("hb") for call in trainer.calls) == expected
