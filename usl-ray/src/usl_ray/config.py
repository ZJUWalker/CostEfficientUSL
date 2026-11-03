"""Configuration loading and validation for all USL-Ray commands."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class RunConfig:
    """Output and reproducibility settings."""

    name: str = "default"
    output_dir: str = "runs"
    mode: Literal["local", "remote", "simulate"] = "local"
    seed: int = 0


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter settings."""

    enabled: bool = False
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")


@dataclass(frozen=True)
class ModelConfig:
    """Model identity and architecture metadata."""

    path: str = ""
    init: Literal["pretrained", "random"] = "pretrained"
    architecture: Literal["qwen3", "llama", "gemma2"] = "qwen3"
    num_hidden_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(frozen=True)
class SplitConfig:
    """Symmetric U-shaped split location."""

    split_point: int = 2


@dataclass(frozen=True)
class BatchConfig:
    """Batching and sequence semantics."""

    batch_size: int = 8
    micro_batch_size: int = 1
    max_seq_len: int = 512
    mask_kind: Literal["causal", "full"] = "causal"
    padding_side: Literal["right", "left"] = "right"
    position_kind: Literal["arange", "from_valid_lens"] = "arange"

    @property
    def num_microbatches(self) -> int:
        """Return the exact number of microbatches in one global batch."""
        return self.batch_size // self.micro_batch_size


@dataclass(frozen=True)
class TrainConfig:
    """Shared client/server optimization settings."""

    steps: int = 20
    epochs: int = 1
    learning_rate: float = 5e-4
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class PipelineConfig:
    """Client and server pipeline schedule."""

    mode: Literal["sequential", "gpipe", "splitmind", "splitmind_eager", "1f1b"] = "splitmind"

    @property
    def server_schedule(self) -> Literal["gpipe", "1f1b"]:
        """Map the client schedule to the torch server schedule."""
        return "1f1b" if self.mode == "1f1b" else "gpipe"


@dataclass(frozen=True)
class NetworkConfig:
    """Expected tc link settings and optional tensor codec."""

    mbps: float = 230.0
    delay_ms: float = 10.0
    compression: Literal["none", "nf4"] = "none"
    timeout_s: float = 600.0


@dataclass(frozen=True)
class ProfileConfig:
    """Runtime profiling settings."""

    enabled: bool = True
    warmup_steps: int = 1
    probe_link: bool = False
    torch_profiler: bool = False


@dataclass(frozen=True)
class ServerConfig:
    """Server-only settings."""

    world_size: int = 1
    backend: Literal["nccl", "gloo"] = "nccl"
    activation_offload_mb: int = 0
    wait_client_s: float = 600.0


@dataclass(frozen=True)
class DataConfig:
    """Client dataset and loss settings."""

    dataset: str = "synthetic"
    path: str = ""
    sample_count: int = 10_000
    shrink_frac: float = 1.0
    sort: Literal["none", "asc", "desc"] = "none"
    loss_on: Literal["all", "completion"] = "all"
    shuffle: bool = False


@dataclass(frozen=True)
class ClientConfig:
    """Client-only settings."""

    data: DataConfig = field(default_factory=DataConfig)
    activation_offload_mb: int = 0
    model_state_offload_layers: int = 0
    wait_server_s: float = 600.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete immutable experiment configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    server: ServerConfig | None = None
    client: ClientConfig | None = None

    @property
    def server_layer_count(self) -> int:
        """Return the decoder layers assigned to the server."""
        return self.model.num_hidden_layers - 2 * self.split.split_point

    def shared_dict(self) -> dict[str, Any]:
        """Return fields that must be identical on client and server."""
        data = asdict(self)
        data.pop("server")
        data.pop("client")
        return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"Cyclic config inheritance: {chain}")
    if not resolved.is_file():
        raise ConfigError(f"Config file does not exist: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {resolved}")
    bases = raw.pop("base", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(item, str) for item in bases):
        raise ConfigError(f"'base' must be a path or list of paths: {resolved}")
    merged: dict[str, Any] = {}
    for base in bases:
        merged = _deep_merge(merged, _load_yaml_tree(resolved.parent / base, (*stack, resolved)))
    return _deep_merge(merged, raw)


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must use key=value syntax: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor: dict[str, Any] = data
        for key in keys[:-1]:
            value = cursor.get(key)
            if not isinstance(value, dict):
                raise ConfigError(f"Unknown override path: {dotted_key}")
            cursor = value
        if keys[-1] not in cursor:
            raise ConfigError(f"Unknown override field: {dotted_key}")
        cursor[keys[-1]] = yaml.safe_load(raw_value)


def _convert(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        return _construct(annotation, value, path)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        options = [item for item in args if item is not type(None)]
        errors: list[str] = []
        for option in options:
            try:
                return _convert(value, option, path)
            except (ConfigError, TypeError, ValueError) as error:
                errors.append(str(error))
        raise ConfigError(f"{path} does not match any allowed type: {errors}")
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path} must be one of {args}, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list")
        item_type = args[0] if args else Any
        return tuple(_convert(item, item_type, f"{path}[{index}]") for index, item in enumerate(value))
    if annotation is Any:
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean")
        return value
    if annotation in (str, int, float):
        if annotation is int and isinstance(value, bool):
            raise ConfigError(f"{path} must be an integer")
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{path} must be {annotation.__name__}, got {value!r}") from error
    return value


def _construct(cls: type[T], data: dict[str, Any], path: str = "config") -> T:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown fields under {path}: {unknown}")
    hints = get_type_hints(cls)
    values = {key: _convert(value, hints[key], f"{path}.{key}") for key, value in data.items()}
    return cls(**values)


def _resolve_model_metadata(config: ExperimentConfig, config_path: Path) -> ExperimentConfig:
    model = config.model
    if model.num_hidden_layers > 0 and model.hidden_size > 0 and model.vocab_size > 0:
        return config
    model_path = Path(model.path)
    if not model_path.is_absolute():
        candidates = [Path.cwd() / model_path, config_path.parent / model_path]
        model_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception as error:
        raise ConfigError(
            "model.num_hidden_layers, model.hidden_size and model.vocab_size must be set "
            f"when model config cannot be loaded from {model_path}"
        ) from error
    layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0))
    hidden = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", 0))
    resolved_model = replace(
        model,
        num_hidden_layers=model.num_hidden_layers or int(layers),
        hidden_size=model.hidden_size or int(hidden),
        vocab_size=model.vocab_size or int(hf_config.vocab_size),
    )
    return replace(config, model=resolved_model)


def validate_config(config: ExperimentConfig, role: Literal["server", "client", "any"] = "any") -> None:
    """Validate all cross-field invariants."""
    if role == "server" and config.server is None:
        raise ConfigError("server section is required for the server command")
    if role == "client" and config.client is None:
        raise ConfigError("client section is required for the client command")
    if config.model.num_hidden_layers <= 0 or config.model.hidden_size <= 0 or config.model.vocab_size <= 0:
        raise ConfigError("model architecture metadata must be positive")
    if config.split.split_point < 1 or 2 * config.split.split_point >= config.model.num_hidden_layers:
        raise ConfigError("split.split_point must leave at least one decoder layer on the server")
    if config.batch.batch_size <= 0 or config.batch.micro_batch_size <= 0:
        raise ConfigError("batch sizes must be positive")
    if config.batch.batch_size % config.batch.micro_batch_size:
        raise ConfigError("batch.batch_size must be divisible by batch.micro_batch_size")
    if config.batch.max_seq_len <= 1:
        raise ConfigError("batch.max_seq_len must be greater than one")
    if config.pipeline.mode == "sequential" and config.batch.micro_batch_size != config.batch.batch_size:
        raise ConfigError("pipeline.mode=sequential requires micro_batch_size == batch_size")
    if config.batch.mask_kind == "full" and config.batch.padding_side == "right":
        pass
    if config.batch.padding_side == "left" and config.batch.position_kind != "from_valid_lens":
        raise ConfigError("left padding requires batch.position_kind=from_valid_lens")
    if config.network.compression == "nf4" and config.model.hidden_size % 64:
        raise ConfigError("NF4 communication requires model.hidden_size divisible by 64")
    if config.train.steps <= 0 or config.train.epochs <= 0 or config.train.learning_rate <= 0:
        raise ConfigError("train steps, epochs and learning rate must be positive")
    if config.profile.warmup_steps < 0 or config.profile.warmup_steps >= config.train.steps:
        raise ConfigError("profile.warmup_steps must be in [0, train.steps)")
    if config.server is not None:
        if config.server.world_size <= 0 or config.server.world_size > config.server_layer_count:
            raise ConfigError("server.world_size must be positive and no larger than the server layer count")
        if config.server.world_size > 1 and config.client is not None and config.client.data.sort != "none":
            raise ConfigError("sorted variable-length batches are unsupported with a multi-stage server")
    if config.client is not None:
        if config.client.model_state_offload_layers > config.split.split_point:
            raise ConfigError("client.model_state_offload_layers cannot exceed split.split_point")
        if config.client.model_state_offload_layers and config.pipeline.mode not in ("splitmind", "gpipe"):
            raise ConfigError("model-state offload is supported only by splitmind and gpipe")
        if (
            config.client.data.sample_count * config.client.data.shrink_frac
            < config.batch.batch_size * config.train.steps
        ):
            raise ConfigError("the client dataset does not contain enough complete batches for train.steps")


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    role: Literal["server", "client", "any"] = "any",
) -> ExperimentConfig:
    """Load, merge, type-check, resolve, and validate one YAML config."""
    config_path = Path(path).expanduser().resolve()
    data = _load_yaml_tree(config_path)
    _apply_overrides(data, overrides or [])
    config = _construct(ExperimentConfig, data)
    config = _resolve_model_metadata(config, config_path)
    validate_config(config, role)
    return config


def shared_config_digest(config: ExperimentConfig) -> str:
    """Return a stable digest used during client/server handshakes."""
    payload = json.dumps(config.shared_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def diff_shared_config(left: ExperimentConfig, right: ExperimentConfig) -> list[str]:
    """Return human-readable differences between shared config fields."""
    differences: list[str] = []

    def visit(prefix: str, first: Any, second: Any) -> None:
        if isinstance(first, dict) and isinstance(second, dict):
            for key in sorted(set(first) | set(second)):
                visit(f"{prefix}.{key}" if prefix else key, first.get(key), second.get(key))
        elif first != second:
            differences.append(f"{prefix}: server={first!r}, client={second!r}")

    visit("", left.shared_dict(), right.shared_dict())
    return differences


def same_experiment(left: ExperimentConfig, right: ExperimentConfig) -> bool:
    """Return whether two configs describe the same shared experiment."""
    return not diff_shared_config(left, right)


def run_tag(config: ExperimentConfig) -> str:
    """Build the sole filesystem-safe experiment identifier."""
    server = config.server
    client = config.client
    parts = [
        f"sp{config.split.split_point}",
        f"b{config.batch.batch_size}",
        f"mb{config.batch.micro_batch_size}",
        f"s{config.batch.max_seq_len}",
        f"ws{server.world_size if server else 0}",
        f"bw{config.network.mbps:g}",
        f"d{config.network.delay_ms:g}",
        config.pipeline.mode,
    ]
    if config.model.lora.enabled:
        parts.append("lora")
    parts.extend(
        (
            f"coa{client.activation_offload_mb if client else 0}",
            f"cos{client.model_state_offload_layers if client else 0}",
            f"soa{server.activation_offload_mb if server else 0}",
        )
    )
    return "_".join(parts)


def result_dir(config: ExperimentConfig) -> Path:
    """Return the deterministic result directory for one experiment."""
    return Path(config.run.output_dir) / config.run.mode / config.run.name / run_tag(config)


def dump_config(config: ExperimentConfig, destination: str | Path) -> None:
    """Persist the fully resolved configuration."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
