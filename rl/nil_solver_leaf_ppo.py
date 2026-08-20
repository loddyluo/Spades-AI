"""Checkpoint and export helpers for four-role Nil solver-leaf PPO."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rl.first4_observation import ENCODER_SCHEMA
from rl.nil_first4_observation import (
    NIL_ENCODER_SCHEMA,
    NilFirstFourFeatureEncoderV1,
)
from rl.nil_solver_leaf_env import NIL_ROLES
from rl.policy_network import PolicyMLP
from rl.solver_leaf_ppo import (
    ACTOR_OUTPUT_DIM,
    TRAINER_SCHEMA,
    PPOConfig,
    ValueMLP,
)


NIL_TRAINER_SCHEMA = "solver-leaf-nil-four-role-ppo-v1"
NIL_ACTOR_SCHEMA = "solver-leaf-nil-role-actor-v1"


@dataclass(frozen=True, slots=True)
class NilTrainingResumeState:
    update: int
    deals_trained: int
    candidate_cursor: int
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NonNilFineTuneSource:
    update: int
    deals_trained: int
    candidate_cursor: int
    config: dict[str, Any]


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_role_objects(mapping: Mapping[str, Any], name: str) -> None:
    if set(mapping) != set(NIL_ROLES):
        raise ValueError(f"{name} must contain exactly the four Nil roles")


def save_nil_training_checkpoint(
    destination: Path,
    actors: Mapping[str, PolicyMLP],
    critics: Mapping[str, ValueMLP],
    optimizers: Mapping[str, torch.optim.Optimizer],
    *,
    update: int,
    deals_trained: int,
    candidate_cursor: int,
    actor_hidden_dims: Sequence[int],
    critic_hidden_dims: Sequence[int],
    ppo_config: PPOConfig,
    run_config: Mapping[str, Any],
) -> None:
    _validate_role_objects(actors, "actors")
    _validate_role_objects(critics, "critics")
    _validate_role_objects(optimizers, "optimizers")
    if any(
        type(value) is not int or value < 0
        for value in (update, deals_trained, candidate_cursor)
    ):
        raise ValueError("checkpoint counters must be nonnegative integers")
    payload = {
        "schema": NIL_TRAINER_SCHEMA,
        "encoder_schema": NIL_ENCODER_SCHEMA,
        "input_dim": NilFirstFourFeatureEncoderV1.TOTAL_DIM,
        "output_dim": ACTOR_OUTPUT_DIM,
        "roles": list(NIL_ROLES),
        "actor_hidden_dims": list(actor_hidden_dims),
        "critic_hidden_dims": list(critic_hidden_dims),
        "role_states": {
            role: {
                "actor_state_dict": actors[role].state_dict(),
                "critic_state_dict": critics[role].state_dict(),
                "optimizer_state_dict": optimizers[role].state_dict(),
            }
            for role in NIL_ROLES
        },
        "update": update,
        "deals_trained": deals_trained,
        "candidate_cursor": candidate_cursor,
        "ppo_config": asdict(ppo_config),
        "run_config": dict(run_config),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    _atomic_torch_save(payload, Path(destination))


def load_nil_training_checkpoint(
    source: Path,
    actors: Mapping[str, PolicyMLP],
    critics: Mapping[str, ValueMLP],
    optimizers: Mapping[str, torch.optim.Optimizer],
    *,
    map_location: torch.device | str = "cpu",
) -> NilTrainingResumeState:
    _validate_role_objects(actors, "actors")
    _validate_role_objects(critics, "critics")
    _validate_role_objects(optimizers, "optimizers")
    payload = torch.load(Path(source), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != NIL_TRAINER_SCHEMA:
        raise ValueError("unsupported four-role Nil trainer checkpoint")
    if payload.get("encoder_schema") != NIL_ENCODER_SCHEMA:
        raise ValueError("Nil trainer checkpoint uses a different encoder schema")
    if payload.get("input_dim") != 536 or payload.get("output_dim") != 52:
        raise ValueError("Nil trainer checkpoint has incompatible dimensions")
    if tuple(payload.get("roles", ())) != NIL_ROLES:
        raise ValueError("Nil trainer checkpoint has incompatible roles")
    role_states = payload.get("role_states")
    if not isinstance(role_states, dict) or set(role_states) != set(NIL_ROLES):
        raise ValueError("Nil trainer checkpoint is missing role states")
    for role in NIL_ROLES:
        state = role_states[role]
        actors[role].load_state_dict(state["actor_state_dict"])
        critics[role].load_state_dict(state["critic_state_dict"])
        optimizers[role].load_state_dict(state["optimizer_state_dict"])
    rng = payload.get("rng", {})
    if set(rng) != {"python", "numpy", "torch"}:
        raise ValueError("Nil trainer checkpoint is missing RNG state")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    return NilTrainingResumeState(
        update=int(payload["update"]),
        deals_trained=int(payload["deals_trained"]),
        candidate_cursor=int(payload["candidate_cursor"]),
        config={
            "actor_hidden_dims": list(payload["actor_hidden_dims"]),
            "critic_hidden_dims": list(payload["critic_hidden_dims"]),
            "ppo_config": dict(payload["ppo_config"]),
            "run_config": dict(payload["run_config"]),
        },
    )


def load_nonnil_finetune_weights(
    source: Path,
    actors: Mapping[str, PolicyMLP],
    critics: Mapping[str, ValueMLP],
    *,
    map_location: torch.device | str = "cpu",
) -> NonNilFineTuneSource:
    """Initialize all roles from one non-Nil trainer while resetting counters."""

    _validate_role_objects(actors, "actors")
    _validate_role_objects(critics, "critics")
    payload = torch.load(Path(source), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != TRAINER_SCHEMA:
        raise ValueError("unsupported non-Nil solver-leaf trainer checkpoint")
    if payload.get("encoder_schema") != ENCODER_SCHEMA:
        raise ValueError("fine-tune source uses a different non-Nil encoder schema")
    if payload.get("input_dim") != 536 or payload.get("output_dim") != 52:
        raise ValueError("fine-tune source has incompatible dimensions")
    for role in NIL_ROLES:
        actors[role].load_state_dict(payload["actor_state_dict"])
        critics[role].load_state_dict(payload["critic_state_dict"])
    return NonNilFineTuneSource(
        update=int(payload["update"]),
        deals_trained=int(payload["deals_trained"]),
        candidate_cursor=int(payload["candidate_cursor"]),
        config={
            "source_schema": TRAINER_SCHEMA,
            "actor_hidden_dims": list(payload["actor_hidden_dims"]),
            "critic_hidden_dims": list(payload["critic_hidden_dims"]),
            "ppo_config": dict(payload["ppo_config"]),
            "run_config": dict(payload["run_config"]),
        },
    )


def load_nil_finetune_weights(
    source: Path,
    actors: Mapping[str, PolicyMLP],
    critics: Mapping[str, ValueMLP],
    *,
    map_location: torch.device | str = "cpu",
) -> NonNilFineTuneSource:
    """Initialize from a previous four-role Nil trainer and reset optimizers."""

    _validate_role_objects(actors, "actors")
    _validate_role_objects(critics, "critics")
    payload = torch.load(Path(source), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != NIL_TRAINER_SCHEMA:
        raise ValueError("unsupported four-role Nil fine-tune checkpoint")
    if payload.get("encoder_schema") != NIL_ENCODER_SCHEMA:
        raise ValueError("Nil fine-tune source uses a different encoder schema")
    if payload.get("input_dim") != 536 or payload.get("output_dim") != 52:
        raise ValueError("Nil fine-tune source has incompatible dimensions")
    if tuple(payload.get("roles", ())) != NIL_ROLES:
        raise ValueError("Nil fine-tune source has incompatible roles")
    role_states = payload.get("role_states")
    if not isinstance(role_states, dict) or set(role_states) != set(NIL_ROLES):
        raise ValueError("Nil fine-tune source is missing role states")
    for role in NIL_ROLES:
        actors[role].load_state_dict(role_states[role]["actor_state_dict"])
        critics[role].load_state_dict(role_states[role]["critic_state_dict"])
    return NonNilFineTuneSource(
        update=int(payload["update"]),
        deals_trained=int(payload["deals_trained"]),
        candidate_cursor=int(payload["candidate_cursor"]),
        config={
            "source_schema": NIL_TRAINER_SCHEMA,
            "actor_hidden_dims": list(payload["actor_hidden_dims"]),
            "critic_hidden_dims": list(payload["critic_hidden_dims"]),
            "ppo_config": dict(payload["ppo_config"]),
            "run_config": dict(payload["run_config"]),
        },
    )


def load_finetune_weights(
    source: Path,
    actors: Mapping[str, PolicyMLP],
    critics: Mapping[str, ValueMLP],
    *,
    map_location: torch.device | str = "cpu",
) -> NonNilFineTuneSource:
    """Load either the compatible non-Nil seed or a prior Nil trainer."""

    payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema == TRAINER_SCHEMA:
        return load_nonnil_finetune_weights(
            source, actors, critics, map_location=map_location
        )
    if schema == NIL_TRAINER_SCHEMA:
        return load_nil_finetune_weights(
            source, actors, critics, map_location=map_location
        )
    raise ValueError("unsupported solver-leaf fine-tune checkpoint")


def export_nil_role_actors(
    save_dir: Path,
    actors: Mapping[str, PolicyMLP],
    *,
    suffix: str,
    actor_hidden_dims: Sequence[int],
    training_update: int,
    deals_trained: int,
    residual_checkpoint_sha256: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export four independently hash-verified actor files plus one manifest."""

    _validate_role_objects(actors, "actors")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    actor_entries: dict[str, Any] = {}
    for role in NIL_ROLES:
        destination = save_dir / f"actor_{role}_{suffix}.pt"
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in actors[role].state_dict().items()
        }
        _atomic_torch_save(state_dict, destination)
        metadata: dict[str, Any] = {
            "schema": NIL_ACTOR_SCHEMA,
            "encoder_schema": NIL_ENCODER_SCHEMA,
            "role": role,
            "input_dim": 536,
            "output_dim": 52,
            "hidden_dims": list(actor_hidden_dims),
            "feature_segments": {
                name: list(bounds)
                for name, bounds in NilFirstFourFeatureEncoderV1.segment_ranges().items()
            },
            "training_update": int(training_update),
            "deals_trained": int(deals_trained),
            "residual_checkpoint_sha256": residual_checkpoint_sha256,
            "actor_sha256": _sha256(destination),
        }
        if extra_metadata:
            metadata["extra"] = dict(extra_metadata)
        sidecar = destination.with_name(f"{destination.name}.json")
        _atomic_json_save(metadata, sidecar)
        actor_entries[role] = {
            "path": destination.name,
            "metadata_path": sidecar.name,
            "sha256": metadata["actor_sha256"],
        }
    manifest = {
        "schema": "solver-leaf-nil-four-role-actor-bundle-v1",
        "encoder_schema": NIL_ENCODER_SCHEMA,
        "roles": list(NIL_ROLES),
        "input_dim": 536,
        "output_dim": 52,
        "training_update": int(training_update),
        "deals_trained": int(deals_trained),
        "actors": actor_entries,
    }
    _atomic_json_save(manifest, save_dir / f"actors_{suffix}.json")
    return manifest


def load_nil_role_actor_bundle(
    manifest_source: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, PolicyMLP], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and verify all four actors referenced by one bundle manifest."""

    manifest_source = Path(manifest_source)
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Nil actor bundle manifest not found: {manifest_source}")
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != (
        "solver-leaf-nil-four-role-actor-bundle-v1"
    ):
        raise ValueError("unsupported Nil actor bundle manifest")
    if manifest.get("encoder_schema") != NIL_ENCODER_SCHEMA:
        raise ValueError("Nil actor bundle uses a different encoder schema")
    if manifest.get("input_dim") != 536 or manifest.get("output_dim") != 52:
        raise ValueError("Nil actor bundle has incompatible dimensions")
    if tuple(manifest.get("roles", ())) != NIL_ROLES:
        raise ValueError("Nil actor bundle has incompatible roles")
    entries = manifest.get("actors")
    if not isinstance(entries, dict) or set(entries) != set(NIL_ROLES):
        raise ValueError("Nil actor bundle is missing role entries")

    actors: dict[str, PolicyMLP] = {}
    metadata_by_role: dict[str, dict[str, Any]] = {}
    hidden_dims: list[int] | None = None
    for role in NIL_ROLES:
        entry = entries[role]
        if not isinstance(entry, dict):
            raise ValueError(f"Nil actor bundle entry {role!r} is invalid")
        actor_path = manifest_source.parent / str(entry.get("path", ""))
        metadata_path = manifest_source.parent / str(entry.get("metadata_path", ""))
        if not actor_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Nil actor and sidecar are required for role {role}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema") != NIL_ACTOR_SCHEMA:
            raise ValueError(f"unsupported Nil actor sidecar for role {role}")
        if metadata.get("encoder_schema") != NIL_ENCODER_SCHEMA:
            raise ValueError(f"Nil actor {role} uses a different encoder schema")
        if metadata.get("role") != role:
            raise ValueError(f"Nil actor sidecar role mismatch for {role}")
        if metadata.get("input_dim") != 536 or metadata.get("output_dim") != 52:
            raise ValueError(f"Nil actor {role} has incompatible dimensions")
        digest = _sha256(actor_path)
        if metadata.get("actor_sha256") != digest or entry.get("sha256") != digest:
            raise ValueError(f"Nil actor SHA-256 mismatch for role {role}")
        role_hidden = metadata.get("hidden_dims")
        if not isinstance(role_hidden, list) or not role_hidden:
            raise ValueError(f"Nil actor {role} sidecar is missing hidden_dims")
        if hidden_dims is None:
            hidden_dims = list(role_hidden)
        elif role_hidden != hidden_dims:
            raise ValueError("Nil bundle actors use different architectures")
        actor = PolicyMLP(input_dim=536, hidden_dims=role_hidden, output_dim=52)
        actor.load_state_dict(
            torch.load(actor_path, map_location=device, weights_only=True)
        )
        actor.to(device)
        actor.eval()
        actors[role] = actor
        metadata_by_role[role] = metadata
    return actors, manifest, metadata_by_role


__all__ = [
    "NIL_ACTOR_SCHEMA",
    "NIL_TRAINER_SCHEMA",
    "NilTrainingResumeState",
    "NonNilFineTuneSource",
    "export_nil_role_actors",
    "load_nil_training_checkpoint",
    "load_nil_role_actor_bundle",
    "load_nil_finetune_weights",
    "load_finetune_weights",
    "load_nonnil_finetune_weights",
    "save_nil_training_checkpoint",
]
