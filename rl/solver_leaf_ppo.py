"""PPO learner and checkpoint helpers for solver-leaf first-four training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from rl.first4_observation import ENCODER_SCHEMA, FirstFourFeatureEncoderV2
from rl.policy_network import PolicyMLP
from rl.solver_leaf_env import LeafTransition, mask_policy_logits


TRAINER_SCHEMA = "solver-leaf-ppo-v1"
ACTOR_OUTPUT_DIM = 52


class ValueMLP(nn.Module):
    """Training-only critic; exported actors do not depend on this module."""

    def __init__(
        self,
        input_dim: int = FirstFourFeatureEncoderV2.TOTAL_DIM,
        hidden_dims: Sequence[int] = (512, 256),
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(type(value) is not int or value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(previous, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        values = self.value_head(self.backbone(x)).squeeze(-1)
        return values.squeeze(0) if squeeze else values


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 1024
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    def __post_init__(self) -> None:
        positive_floats = {
            "learning_rate": self.learning_rate,
            "clip_ratio": self.clip_ratio,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.update_epochs) is not int or self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if type(self.minibatch_size) is not int or self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if not math.isfinite(self.value_coefficient) or self.value_coefficient < 0:
            raise ValueError("value_coefficient must be finite and nonnegative")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    transitions: int
    epochs_completed: int
    minibatches: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    explained_variance: float
    reward_mean: float
    reward_std: float
    advantage_mean_before_normalization: float
    advantage_std_before_normalization: float
    early_stopped_for_kl: bool


@dataclass(frozen=True, slots=True)
class TrainingResumeState:
    update: int
    deals_trained: int
    candidate_cursor: int
    best_validation_margin: float | None
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FineTuneSourceState:
    update: int
    deals_trained: int
    candidate_cursor: int
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TensorTrajectoryBatch:
    features: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


def stack_transitions(
    transitions: Sequence[LeafTransition],
    *,
    device: torch.device | str,
) -> TensorTrajectoryBatch:
    if not transitions:
        raise ValueError("transitions must be nonempty")
    features = np.stack([item.feature for item in transitions]).astype(np.float32, copy=False)
    masks = np.stack([item.legal_mask for item in transitions]).astype(np.bool_, copy=False)
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    old_log_probs = np.asarray([item.old_log_prob for item in transitions], dtype=np.float32)
    returns = np.asarray([item.reward for item in transitions], dtype=np.float32)
    rows = len(transitions)
    if features.shape != (rows, FirstFourFeatureEncoderV2.TOTAL_DIM):
        raise ValueError("transition features must have shape (N, 536)")
    if masks.shape != (rows, ACTOR_OUTPUT_DIM) or not masks.any(axis=1).all():
        raise ValueError("transition legal masks must have shape (N, 52) and be nonempty")
    if not np.isfinite(features).all() or not np.isfinite(old_log_probs).all():
        raise ValueError("transition observations and log probabilities must be finite")
    if not np.isfinite(returns).all():
        raise ValueError("transition returns must be finite")
    if not np.logical_and(actions >= 0, actions < ACTOR_OUTPUT_DIM).all():
        raise ValueError("transition actions must be card ids")
    if not masks[np.arange(rows), actions].all():
        raise ValueError("every transition action must be legal")
    encoded_masks = features[
        :,
        FirstFourFeatureEncoderV2.LEGAL_START : FirstFourFeatureEncoderV2.LEGAL_START
        + ACTOR_OUTPUT_DIM,
    ].astype(np.bool_)
    if not np.array_equal(encoded_masks, masks):
        raise ValueError("transition input masks differ from policy masks")
    resolved_device = torch.device(device)
    return TensorTrajectoryBatch(
        features=torch.from_numpy(features).to(resolved_device),
        legal_masks=torch.from_numpy(masks).to(resolved_device),
        actions=torch.from_numpy(actions).to(resolved_device),
        old_log_probs=torch.from_numpy(old_log_probs).to(resolved_device),
        returns=torch.from_numpy(returns).to(resolved_device),
    )


def policy_log_probs_and_entropy(
    actor: PolicyMLP,
    features: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = actor(features)
    distribution = torch.distributions.Categorical(
        logits=mask_policy_logits(logits, legal_masks)
    )
    return distribution.log_prob(actions), distribution.entropy()


def build_optimizer(
    actor: PolicyMLP,
    critic: ValueMLP,
    config: PPOConfig,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
    )


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    return_variance = torch.var(returns, unbiased=False)
    if float(return_variance.item()) <= 1e-12:
        return 0.0
    residual_variance = torch.var(returns - values, unbiased=False)
    return float((1.0 - residual_variance / return_variance).item())


def ppo_update(
    actor: PolicyMLP,
    critic: ValueMLP,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[LeafTransition],
    config: PPOConfig,
    *,
    device: torch.device | str,
    shuffle_seed: int,
) -> PPOUpdateStats:
    """Apply one PPO update using terminal Monte-Carlo room returns."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be nonnegative")
    batch = stack_transitions(transitions, device=device)
    actor.train()
    critic.train()
    with torch.no_grad():
        old_values = critic(batch.features)
        raw_advantages = batch.returns - old_values
        advantage_mean = raw_advantages.mean()
        advantage_std = raw_advantages.std(unbiased=False)
        if float(advantage_std.item()) > 1e-8:
            advantages = (raw_advantages - advantage_mean) / advantage_std
        else:
            advantages = raw_advantages - advantage_mean

    generator = torch.Generator(device="cpu")
    generator.manual_seed(shuffle_seed)
    total_weight = 0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    kl_sum = 0.0
    clip_fraction_sum = 0.0
    gradient_norm_sum = 0.0
    minibatches = 0
    epochs_completed = 0
    early_stop = False
    parameters = list(actor.parameters()) + list(critic.parameters())

    for _epoch in range(config.update_epochs):
        permutation = torch.randperm(batch.size, generator=generator)
        epoch_kl_weighted = 0.0
        epoch_weight = 0
        for start in range(0, batch.size, config.minibatch_size):
            cpu_indices = permutation[start : start + config.minibatch_size]
            indices = cpu_indices.to(batch.features.device)
            new_log_probs, entropy = policy_log_probs_and_entropy(
                actor,
                batch.features[indices],
                batch.legal_masks[indices],
                batch.actions[indices],
            )
            old_log_probs = batch.old_log_probs[indices]
            minibatch_advantages = advantages[indices]
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            ) * minibatch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            values = critic(batch.features[indices])
            value_loss = torch.nn.functional.mse_loss(values, batch.returns[indices])
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("PPO loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.max_grad_norm
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("PPO gradient norm became non-finite")
            optimizer.step()

            with torch.no_grad():
                approximate_kl = (old_log_probs - new_log_probs).mean()
                clip_fraction = (
                    torch.abs(ratio - 1.0) > config.clip_ratio
                ).to(dtype=torch.float32).mean()
            weight = int(indices.numel())
            total_weight += weight
            epoch_weight += weight
            minibatches += 1
            policy_loss_sum += float(policy_loss.item()) * weight
            value_loss_sum += float(value_loss.item()) * weight
            entropy_sum += float(entropy_mean.item()) * weight
            kl_value = float(approximate_kl.item())
            kl_sum += kl_value * weight
            epoch_kl_weighted += kl_value * weight
            clip_fraction_sum += float(clip_fraction.item()) * weight
            gradient_norm_sum += float(gradient_norm.item()) * weight
        epochs_completed += 1
        if epoch_weight and epoch_kl_weighted / epoch_weight > config.target_kl:
            early_stop = True
            break

    actor.eval()
    critic.eval()
    with torch.no_grad():
        final_values = critic(batch.features)
    if total_weight <= 0:
        raise AssertionError("PPO update did not process any minibatches")
    return PPOUpdateStats(
        transitions=batch.size,
        epochs_completed=epochs_completed,
        minibatches=minibatches,
        policy_loss=policy_loss_sum / total_weight,
        value_loss=value_loss_sum / total_weight,
        entropy=entropy_sum / total_weight,
        approximate_kl=kl_sum / total_weight,
        clip_fraction=clip_fraction_sum / total_weight,
        gradient_norm=gradient_norm_sum / total_weight,
        explained_variance=_explained_variance(batch.returns, final_values),
        reward_mean=float(batch.returns.mean().item()),
        reward_std=float(batch.returns.std(unbiased=False).item()),
        advantage_mean_before_normalization=float(advantage_mean.item()),
        advantage_std_before_normalization=float(advantage_std.item()),
        early_stopped_for_kl=early_stop,
    )


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination = Path(destination)
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
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_checkpoint(
    destination: Path,
    actor: PolicyMLP,
    critic: ValueMLP,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    deals_trained: int,
    candidate_cursor: int,
    best_validation_margin: float | None,
    actor_hidden_dims: Sequence[int],
    critic_hidden_dims: Sequence[int],
    ppo_config: PPOConfig,
    run_config: Mapping[str, Any],
) -> None:
    if any(
        type(value) is not int or value < 0
        for value in (update, deals_trained, candidate_cursor)
    ):
        raise ValueError("checkpoint counters must be nonnegative integers")
    payload = {
        "schema": TRAINER_SCHEMA,
        "encoder_schema": ENCODER_SCHEMA,
        "input_dim": FirstFourFeatureEncoderV2.TOTAL_DIM,
        "output_dim": ACTOR_OUTPUT_DIM,
        "actor_hidden_dims": list(actor_hidden_dims),
        "critic_hidden_dims": list(critic_hidden_dims),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "deals_trained": deals_trained,
        "candidate_cursor": candidate_cursor,
        "best_validation_margin": best_validation_margin,
        "ppo_config": asdict(ppo_config),
        "run_config": dict(run_config),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    _atomic_torch_save(payload, Path(destination))


def load_training_checkpoint(
    source: Path,
    actor: PolicyMLP,
    critic: ValueMLP,
    optimizer: torch.optim.Optimizer,
    *,
    map_location: torch.device | str = "cpu",
) -> TrainingResumeState:
    payload = torch.load(Path(source), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != TRAINER_SCHEMA:
        raise ValueError("unsupported solver-leaf trainer checkpoint")
    if payload.get("encoder_schema") != ENCODER_SCHEMA:
        raise ValueError("trainer checkpoint uses a different encoder schema")
    if payload.get("input_dim") != 536 or payload.get("output_dim") != 52:
        raise ValueError("trainer checkpoint has incompatible actor dimensions")
    actor.load_state_dict(payload["actor_state_dict"])
    critic.load_state_dict(payload["critic_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = payload.get("rng", {})
    if set(rng) != {"python", "numpy", "torch"}:
        raise ValueError("trainer checkpoint is missing RNG state")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    return TrainingResumeState(
        update=int(payload["update"]),
        deals_trained=int(payload["deals_trained"]),
        candidate_cursor=int(payload["candidate_cursor"]),
        best_validation_margin=(
            None
            if payload.get("best_validation_margin") is None
            else float(payload["best_validation_margin"])
        ),
        config={
            "actor_hidden_dims": list(payload["actor_hidden_dims"]),
            "critic_hidden_dims": list(payload["critic_hidden_dims"]),
            "ppo_config": dict(payload["ppo_config"]),
            "run_config": dict(payload["run_config"]),
        },
    )


def load_finetune_weights(
    source: Path,
    actor: PolicyMLP,
    critic: ValueMLP,
    *,
    map_location: torch.device | str = "cpu",
) -> FineTuneSourceState:
    """Load actor/critic weights while deliberately resetting optimizer and counters."""

    payload = torch.load(Path(source), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != TRAINER_SCHEMA:
        raise ValueError("unsupported solver-leaf trainer checkpoint")
    if payload.get("encoder_schema") != ENCODER_SCHEMA:
        raise ValueError("trainer checkpoint uses a different encoder schema")
    if payload.get("input_dim") != 536 or payload.get("output_dim") != 52:
        raise ValueError("trainer checkpoint has incompatible actor dimensions")
    actor.load_state_dict(payload["actor_state_dict"])
    critic.load_state_dict(payload["critic_state_dict"])
    return FineTuneSourceState(
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


def actor_metadata_path(actor_path: Path) -> Path:
    actor_path = Path(actor_path)
    return actor_path.with_name(f"{actor_path.name}.json")


def export_actor(
    destination: Path,
    actor: PolicyMLP,
    *,
    actor_hidden_dims: Sequence[int],
    training_update: int,
    deals_trained: int,
    residual_checkpoint_sha256: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(destination)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in actor.state_dict().items()
    }
    _atomic_torch_save(state_dict, destination)
    metadata: dict[str, Any] = {
        "schema": "solver-leaf-actor-v1",
        "encoder_schema": ENCODER_SCHEMA,
        "input_dim": 536,
        "output_dim": 52,
        "hidden_dims": list(actor_hidden_dims),
        "feature_segments": {
            name: list(bounds)
            for name, bounds in FirstFourFeatureEncoderV2.segment_ranges().items()
        },
        "training_update": int(training_update),
        "deals_trained": int(deals_trained),
        "residual_checkpoint_sha256": residual_checkpoint_sha256,
        "actor_sha256": _sha256(destination),
    }
    if extra_metadata:
        metadata["extra"] = dict(extra_metadata)
    _atomic_json_save(metadata, actor_metadata_path(destination))
    return metadata


def load_exported_actor(
    source: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[PolicyMLP, dict[str, Any]]:
    source = Path(source)
    metadata_file = actor_metadata_path(source)
    if not source.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("actor weights and metadata sidecar are both required")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("encoder_schema") != ENCODER_SCHEMA:
        raise ValueError("actor metadata uses a different encoder schema")
    if metadata.get("input_dim") != 536 or metadata.get("output_dim") != 52:
        raise ValueError("actor metadata has incompatible dimensions")
    if metadata.get("actor_sha256") != _sha256(source):
        raise ValueError("actor checkpoint SHA-256 mismatch")
    hidden_dims = metadata.get("hidden_dims")
    if not isinstance(hidden_dims, list) or not hidden_dims:
        raise ValueError("actor metadata is missing hidden_dims")
    actor = PolicyMLP(input_dim=536, hidden_dims=hidden_dims, output_dim=52)
    state_dict = torch.load(source, map_location=device, weights_only=True)
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor, metadata


__all__ = [
    "ACTOR_OUTPUT_DIM",
    "TRAINER_SCHEMA",
    "PPOConfig",
    "PPOUpdateStats",
    "FineTuneSourceState",
    "TensorTrajectoryBatch",
    "TrainingResumeState",
    "ValueMLP",
    "actor_metadata_path",
    "build_optimizer",
    "export_actor",
    "load_exported_actor",
    "load_finetune_weights",
    "load_training_checkpoint",
    "policy_log_probs_and_entropy",
    "ppo_update",
    "save_training_checkpoint",
    "stack_transitions",
]
