"""Publish or verify a remote experiment contract through the Ray head."""

from __future__ import annotations

import argparse
import time

import ray

ACTOR_NAME = "usl-experiment-contract"
NAMESPACE = "usl-ray"


@ray.remote(num_cpus=0)
class ExperimentContract:
    """Detached signature owned by the server Ray head."""

    def __init__(self, signature: str) -> None:
        self._signature = signature

    def signature(self) -> str:
        """Return the server-declared experiment signature."""
        return self._signature


def _actor(timeout_s: float) -> ray.actor.ActorHandle:
    """Wait for the server contract actor."""
    attempts = max(1, int(timeout_s))
    last_error: ValueError | None = None
    for _ in range(attempts):
        try:
            return ray.get_actor(ACTOR_NAME, namespace=NAMESPACE)
        except ValueError as error:
            last_error = error
            time.sleep(1)
    raise TimeoutError(f"Contract actor was unavailable after {timeout_s:g}s") from last_error


def main() -> int:
    """Publish on the server or verify on the client."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("publish", "verify"))
    parser.add_argument("signature")
    parser.add_argument("--address", required=True)
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args()

    ray.init(address=args.address, namespace=NAMESPACE, ignore_reinit_error=True)
    try:
        if args.action == "publish":
            actor = ExperimentContract.options(name=ACTOR_NAME, lifetime="detached").remote(args.signature)
        else:
            actor = _actor(args.timeout_s)
        actual = ray.get(actor.signature.remote())
        if actual != args.signature:
            raise RuntimeError(f"Remote experiment contract mismatch:\nserver={actual}\nclient={args.signature}")
        return 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
