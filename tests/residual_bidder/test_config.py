from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from residual_bidder.config import BidderConfig, ConfigError, canonical_sha256


BASE_CONFIG = Path("configs/residual_bidder/base.yaml")


def _base_mapping() -> dict[str, object]:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_config(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_base_config_has_frozen_contract() -> None:
    cfg = BidderConfig.load(BASE_CONFIG)

    assert cfg.schema == "stochastic-hybrid-residual-bidder-v1"
    assert (
        cfg.nsfp.sha256
        == "b994add8b3a7067aac95000f8f61f90df9045f5a3d18be5fbc947756a7c2066c"
    )
    assert cfg.play.exact_threshold == 36
    assert cfg.play.enable_nil is True
    assert cfg.play.enable_blind_nil is False
    assert cfg.model.members == 5
    assert cfg.model.input_dim == 167
    assert cfg.model.output_dim == 2
    assert cfg.targets.divisor == 100.0
    assert cfg.workers.outer == 0
    assert cfg.workers.nested_exact == 1


def test_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    raw = _base_mapping()
    targets = raw["targets"]
    assert isinstance(targets, dict)
    targets["clip"] = 5

    with pytest.raises(ConfigError, match=r"unknown key.*targets\.clip"):
        BidderConfig.load(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("policy", "lambda_grid", [-0.1, 0.0], "lambda_grid"),
        ("policy", "temperature_grid", [-0.1, 0.0], "temperature_grid"),
        ("policy", "epsilon_grid", [-0.1, 0.0], "epsilon_grid"),
        ("policy", "epsilon_grid", [0.0, 1.1], "epsilon_grid"),
        ("policy", "rho_grid", [0.0, 0.5], "rho_grid"),
        ("policy", "rho_grid", [0.5, 1.1], "rho_grid"),
    ],
)
def test_config_rejects_out_of_domain_policy_grid(
    tmp_path: Path,
    section: str,
    key: str,
    value: list[float],
    message: str,
) -> None:
    raw = _base_mapping()
    policy = raw[section]
    assert isinstance(policy, dict)
    policy[key] = value

    with pytest.raises(ConfigError, match=message):
        BidderConfig.load(_write_config(tmp_path, raw))


@pytest.mark.parametrize("key", ["lambda_grid", "temperature_grid", "epsilon_grid"])
def test_config_requires_deterministic_grid_point(tmp_path: Path, key: str) -> None:
    raw = _base_mapping()
    policy = raw["policy"]
    assert isinstance(policy, dict)
    policy[key] = [value for value in policy[key] if value != 0.0]

    with pytest.raises(ConfigError, match=r"deterministic grid point"):
        BidderConfig.load(_write_config(tmp_path, raw))


def test_config_rejects_inconsistent_member_seeds(tmp_path: Path) -> None:
    raw = _base_mapping()
    model = raw["model"]
    assert isinstance(model, dict)
    model["init_seeds"] = [1701]

    with pytest.raises(ConfigError, match=r"members.*init_seeds"):
        BidderConfig.load(_write_config(tmp_path, raw))


def test_config_uses_strict_scalar_types(tmp_path: Path) -> None:
    raw = _base_mapping()
    workers = raw["workers"]
    assert isinstance(workers, dict)
    workers["outer"] = False

    with pytest.raises(ConfigError, match=r"workers\.outer.*integer"):
        BidderConfig.load(_write_config(tmp_path, raw))


def test_config_hashes_canonical_json() -> None:
    raw = _base_mapping()
    reordered = {key: copy.deepcopy(raw[key]) for key in reversed(raw)}

    assert canonical_sha256(raw) == canonical_sha256(reordered)
    assert BidderConfig.load(BASE_CONFIG).sha256() == canonical_sha256(raw)
