"""Measure Ray object-transfer bandwidth between USL resource groups."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import ray


@dataclass(frozen=True)
class LinkFit:
    """Linear transfer model using SI Mbit/s."""

    fixed_ms: float
    mbps: float
    samples: dict[int, float]


@ray.remote
class Endpoint:
    """Create or consume objects on one Ray node."""

    def make(self, size: int, nonce: int) -> np.ndarray:
        data = np.zeros(size, dtype=np.uint8)
        data[0] = nonce % 251
        return data

    def consume(self, data: np.ndarray) -> int:
        return int(data.nbytes)


def fit_link(source: ray.actor.ActorHandle, target: ray.actor.ActorHandle, sizes: list[int], repeats: int) -> LinkFit:
    """Fit elapsed milliseconds against bytes using fresh source-owned objects."""
    samples: dict[int, float] = {}
    for size in sizes:
        elapsed: list[float] = []
        for repeat in range(repeats):
            source_ref = source.make.remote(size, repeat)
            started = time.perf_counter()
            received = ray.get(target.consume.remote(source_ref))
            elapsed.append((time.perf_counter() - started) * 1000)
            if received != size:
                raise RuntimeError(f"Expected {size} bytes, received {received}")
        samples[size] = float(np.median(elapsed))
    slope, intercept = np.polyfit(np.asarray(list(samples)), np.asarray(list(samples.values())), 1)
    mbps = 8_000 / slope / 1_000_000
    return LinkFit(max(0.0, float(intercept)), float(mbps), samples)


def main() -> int:
    """Measure both directions and print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-mbps", type=float)
    args = parser.parse_args()
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), namespace="usl-ray", ignore_reinit_error=True)
    server = Endpoint.options(num_cpus=0.1, resources={"usl_server": 0.1}).remote()
    client = Endpoint.options(num_cpus=0.1, resources={"usl_client": 0.1}).remote()
    try:
        sizes = [item * 1024**2 for item in args.sizes_mib]
        result = {
            "uplink": asdict(fit_link(client, server, sizes, args.repeats)),
            "downlink": asdict(fit_link(server, client, sizes, args.repeats)),
        }
        print(json.dumps(result, indent=2))
        if args.expected_mbps:
            for direction, fit in result.items():
                error = abs(fit["mbps"] - args.expected_mbps) / args.expected_mbps
                if error > 0.1:
                    raise RuntimeError(
                        f"{direction} measured {fit['mbps']:.1f} Mbit/s, more than 10% from {args.expected_mbps:g}"
                    )
        return 0
    finally:
        ray.kill(server)
        ray.kill(client)
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
