"""Strict, immutable configuration for the residual bidder pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a bidder configuration violates its frozen schema."""


def canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NSFPConfig:
    path: str
    sha256: str


@dataclass(frozen=True)
class PlayConfig:
    config_path: str
    config_sha256: str
    source_manifest: tuple[str, ...]
    exact_threshold: int
    first_tricks: int
    enable_nil: bool
    enable_blind_nil: bool


@dataclass(frozen=True)
class TargetConfig:
    divisor: float


@dataclass(frozen=True)
class ModelConfig:
    members: int
    input_dim: int
    hidden_dim: int
    bottleneck_dim: int
    output_dim: int
    margin_divisor: float
    init_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PolicyGridConfig:
    policy_seed: int
    lambda_grid: tuple[float, ...]
    temperature_grid: tuple[float, ...]
    epsilon_grid: tuple[float, ...]
    rho_grid: tuple[float, ...]


@dataclass(frozen=True)
class WorkerConfig:
    outer: int
    nested_exact: int


@dataclass(frozen=True)
class DataConfig:
    block_deals: int
    shard_deals: int
    natural_fraction: float
    reservoir_fraction: float
    reservoir_capacity_per_stratum: int
    fixed_probe_states: int
    minimum_blocks: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stop_patience: int
    gradient_norm_clip: float
    precision: str


@dataclass(frozen=True)
class CalibrationConfig:
    halving_eta: int
    round_deals: tuple[int, ...]
    real_play_shortlist: int
    real_play_deals: int


@dataclass(frozen=True)
class PromotionConfig:
    alpha_one_sided: float
    power: float
    minimum_detectable_points: float
    minimum_deals: int
    round_to_deals: int
    bootstrap_resamples: int
    protected_stratum_fraction: float
    protected_behavior_strata: tuple[str, ...]


@dataclass(frozen=True)
class StoppingConfig:
    latest_blocks: int
    probability_l1: float
    changed_probe_fraction: float


@dataclass(frozen=True)
class StorageConfig:
    run_dir: str


@dataclass(frozen=True)
class BidderConfig:
    schema: str
    nsfp: NSFPConfig
    play: PlayConfig
    targets: TargetConfig
    model: ModelConfig
    policy: PolicyGridConfig
    workers: WorkerConfig
    data: DataConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    promotion: PromotionConfig
    stopping: StoppingConfig
    storage: StorageConfig

    @classmethod
    def load(cls, path: Path) -> BidderConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"cannot load config {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ConfigError("config root must be a mapping")
        config = _decode_dataclass(cls, raw, "")
        config._validate()
        return config

    def sha256(self) -> str:
        return canonical_sha256(asdict(self))

    def _validate(self) -> None:
        if self.schema != "stochastic-hybrid-residual-bidder-v1":
            raise ConfigError(f"unsupported schema: {self.schema!r}")
        _validate_hash("nsfp.sha256", self.nsfp.sha256)
        _validate_hash("play.config_sha256", self.play.config_sha256)
        _nonempty("nsfp.path", self.nsfp.path)
        _nonempty("play.config_path", self.play.config_path)
        _nonempty("storage.run_dir", self.storage.run_dir)
        if not self.play.source_manifest or any(not item for item in self.play.source_manifest):
            raise ConfigError("play.source_manifest must contain non-empty paths")
        _positive_int("play.exact_threshold", self.play.exact_threshold)
        if not 0 <= self.play.first_tricks <= 13:
            raise ConfigError("play.first_tricks must be between 0 and 13")
        _positive("targets.divisor", self.targets.divisor)

        for name in ("members", "input_dim", "hidden_dim", "bottleneck_dim", "output_dim"):
            _positive_int(f"model.{name}", getattr(self.model, name))
        _positive("model.margin_divisor", self.model.margin_divisor)
        if len(self.model.init_seeds) != self.model.members:
            raise ConfigError("model.members must equal the number of model.init_seeds")
        if len(set(self.model.init_seeds)) != len(self.model.init_seeds):
            raise ConfigError("model.init_seeds must be unique")

        _grid("policy.lambda_grid", self.policy.lambda_grid, lambda value: value >= 0)
        _grid("policy.temperature_grid", self.policy.temperature_grid, lambda value: value >= 0)
        _grid("policy.epsilon_grid", self.policy.epsilon_grid, lambda value: 0 <= value <= 1)
        _grid("policy.rho_grid", self.policy.rho_grid, lambda value: 0 < value <= 1)
        if not all(
            0.0 in grid
            for grid in (
                self.policy.lambda_grid,
                self.policy.temperature_grid,
                self.policy.epsilon_grid,
            )
        ):
            raise ConfigError(
                "policy grids must include the deterministic grid point "
                "lambda=0, temperature=0, epsilon=0"
            )

        if self.workers.outer < 0:
            raise ConfigError("workers.outer must be non-negative")
        _positive_int("workers.nested_exact", self.workers.nested_exact)

        for name in (
            "block_deals",
            "shard_deals",
            "reservoir_capacity_per_stratum",
            "fixed_probe_states",
            "minimum_blocks",
        ):
            _positive_int(f"data.{name}", getattr(self.data, name))
        _fraction("data.natural_fraction", self.data.natural_fraction)
        _fraction("data.reservoir_fraction", self.data.reservoir_fraction)
        if not math.isclose(
            self.data.natural_fraction + self.data.reservoir_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConfigError("data natural_fraction and reservoir_fraction must sum to 1")
        if self.data.block_deals % self.data.shard_deals:
            raise ConfigError("data.block_deals must be divisible by data.shard_deals")

        for name in ("batch_size", "max_epochs", "early_stop_patience"):
            _positive_int(f"training.{name}", getattr(self.training, name))
        _positive("training.learning_rate", self.training.learning_rate)
        _nonnegative("training.weight_decay", self.training.weight_decay)
        _positive("training.gradient_norm_clip", self.training.gradient_norm_clip)
        if self.training.precision != "float32":
            raise ConfigError("training.precision must be 'float32'")

        if self.calibration.halving_eta <= 1:
            raise ConfigError("calibration.halving_eta must be greater than 1")
        _strictly_increasing_positive("calibration.round_deals", self.calibration.round_deals)
        _positive_int("calibration.real_play_shortlist", self.calibration.real_play_shortlist)
        _positive_int("calibration.real_play_deals", self.calibration.real_play_deals)

        _open_fraction("promotion.alpha_one_sided", self.promotion.alpha_one_sided)
        _open_fraction("promotion.power", self.promotion.power)
        _positive("promotion.minimum_detectable_points", self.promotion.minimum_detectable_points)
        for name in ("minimum_deals", "round_to_deals", "bootstrap_resamples"):
            _positive_int(f"promotion.{name}", getattr(self.promotion, name))
        if self.promotion.minimum_deals % self.promotion.round_to_deals:
            raise ConfigError("promotion.minimum_deals must be divisible by round_to_deals")
        _open_fraction(
            "promotion.protected_stratum_fraction",
            self.promotion.protected_stratum_fraction,
        )
        if not self.promotion.protected_behavior_strata:
            raise ConfigError("promotion.protected_behavior_strata must not be empty")

        _positive_int("stopping.latest_blocks", self.stopping.latest_blocks)
        _nonnegative("stopping.probability_l1", self.stopping.probability_l1)
        _fraction("stopping.changed_probe_fraction", self.stopping.changed_probe_fraction)


def _decode_dataclass(cls: type[Any], raw: Mapping[str, object], path: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    type_hints = get_type_hints(cls)
    field_names = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        location = f"{path}.{unknown[0]}" if path else unknown[0]
        raise ConfigError(f"unknown key {location}")
    missing = sorted(field_names - set(raw))
    if missing:
        location = f"{path}.{missing[0]}" if path else missing[0]
        raise ConfigError(f"missing key {location}")
    values = {}
    for field in fields(cls):
        location = f"{path}.{field.name}" if path else field.name
        values[field.name] = _decode_value(type_hints[field.name], raw[field.name], location)
    return cls(**values)


def _decode_value(expected: Any, value: object, path: str) -> Any:
    origin = get_origin(expected)
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        item_type = get_args(expected)[0]
        return tuple(_decode_value(item_type, item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(expected, type) and is_dataclass(expected):
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        return _decode_dataclass(expected, value, path)
    if expected is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path} must be a boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise ConfigError(f"{path} must be an integer")
        return value
    if expected is float:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ConfigError(f"{path} must be a finite number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    if origin in (Union, types.UnionType):
        for option in get_args(expected):
            try:
                return _decode_value(option, value, path)
            except ConfigError:
                pass
    raise ConfigError(f"unsupported type for {path}: {expected}")


def _validate_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ConfigError(f"{name} must be a lowercase SHA-256 digest")


def _nonempty(name: str, value: str) -> None:
    if not value:
        raise ConfigError(f"{name} must not be empty")


def _positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ConfigError(f"{name} must be non-negative")


def _fraction(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ConfigError(f"{name} must be between 0 and 1")


def _open_fraction(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ConfigError(f"{name} must be strictly between 0 and 1")


def _grid(name: str, values: tuple[float, ...], valid: Any) -> None:
    if not values or any(not valid(value) for value in values):
        raise ConfigError(f"{name} contains an out-of-domain value")
    if tuple(sorted(set(values))) != values:
        raise ConfigError(f"{name} must be strictly increasing without duplicates")


def _strictly_increasing_positive(name: str, values: tuple[int, ...]) -> None:
    if not values or any(value <= 0 for value in values) or tuple(sorted(set(values))) != values:
        raise ConfigError(f"{name} must be strictly increasing positive integers")
