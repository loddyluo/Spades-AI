"""Train four role-specific exactly-one-Nil solver-leaf PPO policies."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from residual_bidder.deployment import DEPLOYED_CHECKPOINT_SHA256
from rl.nil_first4_observation import (
    NIL_ENCODER_SCHEMA,
    NilFirstFourFeatureEncoderV1,
)
from rl.nil_solver_leaf_env import (
    NIL_ROLES,
    NilCollectionBatch,
    NilProductionDuplicateCollector,
)
from rl.nil_solver_leaf_ppo import (
    export_nil_role_actors,
    load_finetune_weights,
    load_nil_training_checkpoint,
    save_nil_training_checkpoint,
)
from rl.policy_network import PolicyMLP
from rl.solver_leaf_env import OpponentPoolConfig
from rl.solver_leaf_ppo import PPOConfig, ValueMLP, build_optimizer, ppo_update


DEFAULT_ACTOR_HIDDEN_DIMS = (1024, 512, 512)
DEFAULT_CRITIC_HIDDEN_DIMS = (512, 256)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-role 536-d exactly-one-Nil solver-leaf PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--total-games", type=int, default=100_000)
    parser.add_argument("--rollout-deals", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=636_042)
    parser.add_argument("--base-shuffle-seed", type=int, default=63_600_000)
    parser.add_argument("--bid-policy-seed", type=int, default=None)
    parser.add_argument(
        "--actor-hidden-dims", type=int, nargs="+", default=list(DEFAULT_ACTOR_HIDDEN_DIMS)
    )
    parser.add_argument(
        "--critic-hidden-dims", type=int, nargs="+", default=list(DEFAULT_CRITIC_HIDDEN_DIMS)
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.003)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--save-dir", default="output/solver-leaf-nil-four-role-phase1"
    )
    parser.add_argument("--save-every-updates", type=int, default=5)
    parser.add_argument("--oversample-factor", type=float, default=7.0)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None)
    initialization.add_argument("--finetune-from", default=None)
    parser.add_argument("--rule-opponent-weight", type=float, default=1.0)
    parser.add_argument("--champion-opponent-weight", type=float, default=0.0)
    parser.add_argument("--history-opponent-weight", type=float, default=0.0)
    parser.add_argument("--champion-checkpoint", default=None)
    parser.add_argument("--history-checkpoints", nargs="*", default=[])
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if type(args.total_games) is not int or args.total_games <= 0 or args.total_games % 2:
        raise ValueError("--total-games must be a positive even integer")
    for name in ("rollout_deals", "workers", "save_every_updates"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("seed", "base_shuffle_seed"):
        value = getattr(args, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if any(value <= 0 for value in args.actor_hidden_dims + args.critic_hidden_dims):
        raise ValueError("hidden dimensions must be positive")
    if not math.isfinite(args.oversample_factor) or args.oversample_factor < 1.0:
        raise ValueError("--oversample-factor must be finite and at least one")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    if args.finetune_from and not Path(args.finetune_from).is_file():
        raise FileNotFoundError(f"fine-tune trainer checkpoint not found: {args.finetune_from}")
    opponent_pool = _opponent_pool_config(args)
    checkpoints: list[str] = []
    if opponent_pool.champion_weight > 0.0:
        assert opponent_pool.champion_checkpoint is not None
        checkpoints.append(opponent_pool.champion_checkpoint)
    if opponent_pool.history_weight > 0.0:
        checkpoints.extend(opponent_pool.history_checkpoints)
    for checkpoint in checkpoints:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Nil opponent bundle manifest not found: {checkpoint}")


def _ppo_config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )


def _opponent_pool_config(args: argparse.Namespace) -> OpponentPoolConfig:
    return OpponentPoolConfig(
        rule_weight=args.rule_opponent_weight,
        champion_weight=args.champion_opponent_weight,
        history_weight=args.history_opponent_weight,
        champion_checkpoint=args.champion_checkpoint,
        history_checkpoints=tuple(args.history_checkpoints),
    )


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_resume_settings(
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    saved: dict[str, Any],
) -> None:
    if saved.get("actor_hidden_dims") != list(args.actor_hidden_dims):
        raise ValueError("resume checkpoint actor architecture differs from CLI")
    if saved.get("critic_hidden_dims") != list(args.critic_hidden_dims):
        raise ValueError("resume checkpoint critic architecture differs from CLI")
    if saved.get("ppo_config") != asdict(ppo_config):
        raise ValueError("resume checkpoint PPO configuration differs from CLI")
    saved_run = saved.get("run_config")
    if not isinstance(saved_run, dict):
        raise ValueError("resume checkpoint is missing its run configuration")
    legacy_defaults = {
        "rule_opponent_weight": 1.0,
        "champion_opponent_weight": 0.0,
        "history_opponent_weight": 0.0,
        "champion_checkpoint": None,
        "history_checkpoints": [],
    }
    for key in (
        "seed",
        "base_shuffle_seed",
        "bid_policy_seed",
        "rollout_deals",
        "oversample_factor",
        "rule_opponent_weight",
        "champion_opponent_weight",
        "history_opponent_weight",
        "champion_checkpoint",
        "history_checkpoints",
    ):
        if saved_run.get(key, legacy_defaults.get(key)) != getattr(args, key):
            raise ValueError(
                f"resume checkpoint {key!r} differs from CLI; exact continuation refused"
            )


def _collection_metrics(batch: NilCollectionBatch) -> dict[str, Any]:
    latencies = np.asarray(
        [
            room.solver_seconds
            for deal in batch.deals
            for room in (deal.room_team0, deal.room_team1)
        ],
        dtype=np.float64,
    )
    margins = [deal.duplicate_margin_points for deal in batch.deals]
    role_counts = {
        role: len(transitions)
        for role, transitions in batch.transitions_by_role.items()
    }
    accepted = len(batch.deals)
    opponent_counts = Counter(deal.opponent_id for deal in batch.deals)
    return {
        "duplicate_deals": accepted,
        "games": accepted * 2,
        "solver_calls": batch.solver_calls,
        "scanned_candidates": batch.scanned_candidates,
        "nil_count_histogram": {
            str(key): value for key, value in batch.nil_count_histogram.items()
        },
        "exactly_one_nil_rate": accepted / batch.scanned_candidates,
        "elapsed_seconds": batch.elapsed_seconds,
        "accepted_games_per_second": accepted * 2 / batch.elapsed_seconds,
        "mean_duplicate_margin_points": statistics.fmean(margins),
        "role_transition_counts": role_counts,
        "opponent_deal_counts": dict(sorted(opponent_counts.items())),
        "opponent_deal_fractions": {
            key: count / accepted for key, count in sorted(opponent_counts.items())
        },
        "solver_seconds_p50": float(np.percentile(latencies, 50)),
        "solver_seconds_p95": float(np.percentile(latencies, 95)),
        "solver_seconds_max": float(latencies.max()),
        "worker_peak_rss_bytes": batch.worker_peak_rss_bytes,
        "aggregate_worker_peak_rss_bytes": batch.aggregate_worker_peak_rss_bytes,
    }


def _save_artifacts(
    args: argparse.Namespace,
    actors: dict[str, PolicyMLP],
    critics: dict[str, ValueMLP],
    optimizers: dict[str, torch.optim.Optimizer],
    ppo_config: PPOConfig,
    *,
    update: int,
    deals_trained: int,
    candidate_cursor: int,
    suffix: str,
    run_config: dict[str, Any],
) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_nil_training_checkpoint(
        save_dir / f"trainer_{suffix}.pt",
        actors,
        critics,
        optimizers,
        update=update,
        deals_trained=deals_trained,
        candidate_cursor=candidate_cursor,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        ppo_config=ppo_config,
        run_config=run_config,
    )
    export_nil_role_actors(
        save_dir,
        actors,
        suffix=suffix,
        actor_hidden_dims=args.actor_hidden_dims,
        training_update=update,
        deals_trained=deals_trained,
        residual_checkpoint_sha256=DEPLOYED_CHECKPOINT_SHA256,
        extra_metadata={
            "ppo_config": asdict(ppo_config),
            "training_config": run_config,
        },
    )


def train(args: argparse.Namespace) -> None:
    _set_seeds(args.seed)
    device = torch.device(args.device)
    actors = {
        role: PolicyMLP(
            input_dim=NilFirstFourFeatureEncoderV1.TOTAL_DIM,
            hidden_dims=args.actor_hidden_dims,
            output_dim=52,
        ).to(device)
        for role in NIL_ROLES
    }
    critics = {
        role: ValueMLP(536, args.critic_hidden_dims).to(device) for role in NIL_ROLES
    }
    ppo_config = _ppo_config(args)
    optimizers = {
        role: build_optimizer(actors[role], critics[role], ppo_config)
        for role in NIL_ROLES
    }
    update = 0
    deals_trained = 0
    candidate_cursor = 0
    artifact_run_config = vars(args).copy()
    finetune_source: dict[str, Any] | None = None

    if args.finetune_from:
        source = load_finetune_weights(
            Path(args.finetune_from), actors, critics, map_location=device
        )
        if source.config["actor_hidden_dims"] != list(args.actor_hidden_dims):
            raise ValueError("fine-tune checkpoint actor architecture differs from CLI")
        if source.config["critic_hidden_dims"] != list(args.critic_hidden_dims):
            raise ValueError("fine-tune checkpoint critic architecture differs from CLI")
        finetune_source = {
            "path": str(Path(args.finetune_from)),
            "source_trainer_schema": source.config["source_schema"],
            "source_update": source.update,
            "source_deals_trained": source.deals_trained,
            "optimizer_and_counters_reset": True,
        }

    if args.resume:
        resume = load_nil_training_checkpoint(
            Path(args.resume), actors, critics, optimizers, map_location=device
        )
        _validate_resume_settings(args, ppo_config, resume.config)
        update = resume.update
        deals_trained = resume.deals_trained
        candidate_cursor = resume.candidate_cursor
        artifact_run_config = dict(resume.config["run_config"])
        artifact_run_config.update(
            {
                "resume": str(Path(args.resume)),
                "total_games": args.total_games,
                "save_dir": args.save_dir,
                "save_every_updates": args.save_every_updates,
                "workers": args.workers,
                "device": args.device,
            }
        )

    target_deals = args.total_games // 2
    if deals_trained > target_deals:
        raise ValueError("resume checkpoint already exceeds --total-games")
    print(
        json.dumps(
            {
                "training_start": {
                    "trainer_schema": "solver-leaf-nil-four-role-ppo-v1",
                    "encoder_schema": NIL_ENCODER_SCHEMA,
                    "roles": list(NIL_ROLES),
                    "input_dim": 536,
                    "output_dim": 52,
                    "target_games": args.total_games,
                    "target_duplicate_deals": target_deals,
                    "workers": args.workers,
                    "device": str(device),
                    "residual_checkpoint_sha256": DEPLOYED_CHECKPOINT_SHA256,
                    "finetune_source": finetune_source,
                    "opponent_pool": asdict(_opponent_pool_config(args)),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    wall_started = time.perf_counter()
    with NilProductionDuplicateCollector(
        workers=args.workers,
        actor_hidden_dims=args.actor_hidden_dims,
        bid_policy_seed=args.bid_policy_seed,
        opponent_pool_config=_opponent_pool_config(args),
        oversample_factor=args.oversample_factor,
    ) as collector:
        while deals_trained < target_deals:
            rollout_deals = min(args.rollout_deals, target_deals - deals_trained)
            collection = collector.collect(
                actors,
                start_candidate_index=candidate_cursor,
                target_deals=rollout_deals,
                base_shuffle_seed=args.base_shuffle_seed,
                run_seed=args.seed,
                deterministic=False,
                record_transitions=True,
            )
            candidate_cursor = collection.next_candidate_index
            transitions_by_role = collection.transitions_by_role
            role_stats: dict[str, Any] = {}
            ppo_started = time.perf_counter()
            for role_index, role in enumerate(NIL_ROLES):
                expected = len(collection.deals) * 4
                transitions = transitions_by_role[role]
                if len(transitions) != expected:
                    raise AssertionError(
                        f"role {role} got {len(transitions)} transitions, expected {expected}"
                    )
                stats = ppo_update(
                    actors[role],
                    critics[role],
                    optimizers[role],
                    transitions,
                    ppo_config,
                    device=device,
                    shuffle_seed=args.seed + update * len(NIL_ROLES) + role_index,
                )
                role_stats[role] = asdict(stats)
            ppo_seconds = time.perf_counter() - ppo_started
            deals_trained += len(collection.deals)
            update += 1
            metrics = _collection_metrics(collection)
            scalar_fields = (
                "policy_loss",
                "value_loss",
                "entropy",
                "approximate_kl",
                "clip_fraction",
                "gradient_norm",
            )
            role_means = {
                field: statistics.fmean(
                    float(role_stats[role][field]) for role in NIL_ROLES
                )
                for field in scalar_fields
            }
            wall_elapsed = time.perf_counter() - wall_started
            games_trained = deals_trained * 2
            cumulative_rate = games_trained / wall_elapsed
            remaining_games = args.total_games - games_trained
            metrics.update(
                {
                    "update": update,
                    "deals_trained": deals_trained,
                    "games_trained": games_trained,
                    "candidate_cursor": candidate_cursor,
                    "ppo_seconds": ppo_seconds,
                    "ppo_by_role": role_stats,
                    "ppo_role_means": role_means,
                    "cumulative_accepted_games_per_second": cumulative_rate,
                    "eta_seconds": remaining_games / cumulative_rate,
                }
            )
            print(json.dumps({"training_update": metrics}, ensure_ascii=False), flush=True)
            if update % args.save_every_updates == 0:
                _save_artifacts(
                    args,
                    actors,
                    critics,
                    optimizers,
                    ppo_config,
                    update=update,
                    deals_trained=deals_trained,
                    candidate_cursor=candidate_cursor,
                    suffix=f"update_{update:06d}",
                    run_config=artifact_run_config,
                )

    _save_artifacts(
        args,
        actors,
        critics,
        optimizers,
        ppo_config,
        update=update,
        deals_trained=deals_trained,
        candidate_cursor=candidate_cursor,
        suffix="final",
        run_config=artifact_run_config,
    )
    print(
        json.dumps(
            {
                "training_complete": {
                    "updates": update,
                    "games_trained": deals_trained * 2,
                    "save_dir": str(Path(args.save_dir)),
                    "roles": list(NIL_ROLES),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    train(args)


if __name__ == "__main__":
    main()
