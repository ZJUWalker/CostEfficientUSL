"""Command-line entry points for separate client and server processes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import ray

from usl_ray.client.launcher import run_client
from usl_ray.config import ExperimentConfig, load_config
from usl_ray.profile.plot import plot_profile, plot_timelines
from usl_ray.server.launcher import RAY_NAMESPACE, run_server
from usl_ray.simulate.cost_model import CostModel, MainVariable, fit_from_profiles
from usl_ray.simulate.search import compare_profile, pareto_search, save_search
from usl_ray.simulate.simulator import simulate_splitmind


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Role-specific YAML configuration")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Typed config override")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config without starting Ray")


def _load(args: argparse.Namespace, role: Literal["server", "client"]) -> ExperimentConfig:
    config = load_config(args.config, args.set, role=role)
    if args.dry_run:
        print(json.dumps(asdict(config), indent=2))
    return config


def _connect_ray() -> None:
    address = os.environ.get("RAY_ADDRESS", "auto")
    ray.init(address=address, namespace=RAY_NAMESPACE, ignore_reinit_error=True)


def _server(args: argparse.Namespace) -> int:
    config = _load(args, "server")
    if args.dry_run:
        return 0
    _connect_ray()
    try:
        results = run_server(config)
        print(
            json.dumps(
                [{"rank": item.rank, "max_memory_mib": item.max_memory_mib} for item in results],
                indent=2,
            )
        )
        return 0
    finally:
        ray.shutdown()


def _client(args: argparse.Namespace) -> int:
    config = _load(args, "client")
    if args.dry_run:
        return 0
    _connect_ray()
    try:
        result = run_client(config, num_gpus=float(os.environ.get("USL_CLIENT_GPUS", "1")))
        print(
            json.dumps(
                {
                    "losses": result.losses,
                    "max_memory_mib": result.max_memory_mib,
                    "server_memory_mib": result.server_memory_mib,
                },
                indent=2,
            )
        )
        return 0
    finally:
        ray.shutdown()


def _variables(config: ExperimentConfig) -> MainVariable:
    if config.client is None:
        raise ValueError("Simulation requires a client section")
    return MainVariable(
        total_sample_count=config.client.data.sample_count,
        batch_size=config.batch.batch_size,
        split_point=config.split.split_point,
        client_offload_mb_num=config.client.activation_offload_mb,
        server_offload_mb_num=config.server.activation_offload_mb if config.server else 0,
        client_offload_model_state_sp_num=config.client.model_state_offload_layers,
        lora=config.model.lora.enabled,
    )


def _simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set, role="any")
    model = CostModel.load(args.cost_model)
    result = simulate_splitmind(_variables(config), model.time, model.memory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"variables": asdict(result.variables), "objective": asdict(result.objective)}, indent=2),
        encoding="utf-8",
    )
    plot_timelines((result.timeline,), output.with_suffix(".png"))
    print(json.dumps(asdict(result.objective), indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    comparison = compare_profile(args.profile, CostModel.load(args.cost_model))
    print(json.dumps(asdict(comparison), indent=2))
    return 0


def _search(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set, role="any")
    model = CostModel.load(args.cost_model)
    results = pareto_search(
        _variables(config),
        model,
        max_batch_size=args.max_batch_size,
        max_client_memory_mib=args.max_client_memory_mib,
    )
    save_search(results, args.output)
    print(f"Saved {len(results)} Pareto points to {args.output}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    profiles: dict[tuple[int, int, int, int, int], dict[str, object]] = {}
    for profile_path in args.profiles:
        payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        config = payload["config"]
        key = (
            config["split"]["split_point"],
            config["batch"]["batch_size"],
            config["client"]["activation_offload_mb"],
            config["client"]["model_state_offload_layers"],
            config["server"]["activation_offload_mb"],
        )
        profiles[key] = payload
    model = fit_from_profiles(
        profiles,
        base_split_point=args.base_split_point,
        base_batch_size=args.base_batch_size,
        max_split_point=args.max_split_point,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        lora=args.lora,
    )
    model.save(args.output)
    print(args.output)
    return 0


def _plot(args: argparse.Namespace) -> int:
    print(
        plot_profile(
            args.profile,
            args.output,
            all_measured=args.all_measured,
            include_warmup=args.include_warmup,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the public CLI parser."""
    parser = argparse.ArgumentParser(prog="usl-ray")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("server", help="Run server pipeline actors")
    _common_parser(server)
    server.set_defaults(handler=_server)
    client = commands.add_parser("client", help="Run the data-owning client actor")
    _common_parser(client)
    client.set_defaults(handler=_client)
    simulate = commands.add_parser("simulate", help="Simulate one YAML configuration")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--cost-model", required=True)
    simulate.add_argument("--output", default="runs/simulation.json")
    simulate.add_argument("--set", action="append", default=[])
    simulate.set_defaults(handler=_simulate)
    compare = commands.add_parser("compare", help="Compare one profile with simulation")
    compare.add_argument("--profile", required=True)
    compare.add_argument("--cost-model", required=True)
    compare.set_defaults(handler=_compare)
    search = commands.add_parser("search", help="Search a Pareto front")
    search.add_argument("--config", required=True)
    search.add_argument("--cost-model", required=True)
    search.add_argument("--max-batch-size", type=int, required=True)
    search.add_argument("--max-client-memory-mib", type=float, required=True)
    search.add_argument("--output", default="runs/pareto.json")
    search.add_argument("--set", action="append", default=[])
    search.set_defaults(handler=_search)
    calibrate = commands.add_parser("calibrate", help="Fit a cost model from seven profiles")
    calibrate.add_argument("profiles", nargs="+")
    calibrate.add_argument("--base-split-point", type=int, required=True)
    calibrate.add_argument("--base-batch-size", type=int, required=True)
    calibrate.add_argument("--max-split-point", type=int, required=True)
    calibrate.add_argument("--micro-batch-size", type=int, default=1)
    calibrate.add_argument("--max-seq-len", type=int, default=512)
    calibrate.add_argument("--lora", action="store_true")
    calibrate.add_argument("--output", default="runs/cost_model.json")
    calibrate.set_defaults(handler=_calibrate)
    plot = commands.add_parser("plot", help="Plot one runtime profile")
    plot.add_argument("profile")
    plot.add_argument("--output")
    plot.add_argument("--all-measured", action="store_true", help="Plot every measured step instead of only the last")
    plot.add_argument("--include-warmup", action="store_true", help="Include warmup timelines")
    plot.set_defaults(handler=_plot)
    return parser


def main() -> int:
    """Run one CLI command."""
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
