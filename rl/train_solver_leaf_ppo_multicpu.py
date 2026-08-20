"""Train or benchmark the 536-d first-four solver-leaf PPO policy."""

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
from rl.first4_observation import ENCODER_SCHEMA, FirstFourFeatureEncoderV2
from rl.policy_network import PolicyMLP
from rl.solver_leaf_env import (
    CollectionBatch,
    OpponentPoolConfig,
    ProductionDuplicateCollector,
)
from rl.solver_leaf_ppo import (
    PPOConfig,
    ValueMLP,
    build_optimizer,
    export_actor,
    load_finetune_weights,
    load_training_checkpoint,
    ppo_update,
    save_training_checkpoint,
)


DEFAULT_ACTOR_HIDDEN_DIMS = (1024, 512, 512)
DEFAULT_CRITIC_HIDDEN_DIMS = (512, 256)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="536-d first-four PPO with one exact solver leaf per room",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--total-games", type=int, default=500_000)
    parser.add_argument("--rollout-deals", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=536_042)
    parser.add_argument("--base-shuffle-seed", type=int, default=53_600_000)
    parser.add_argument("--bid-policy-seed", type=int, default=None)
    parser.add_argument(
        "--actor-hidden-dims", type=int, nargs="+", default=list(DEFAULT_ACTOR_HIDDEN_DIMS)
    )
    parser.add_argument(
        "--critic-hidden-dims", type=int, nargs="+", default=list(DEFAULT_CRITIC_HIDDEN_DIMS)
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="output/solver-leaf-ppo")
    parser.add_argument("--save-every-updates", type=int, default=10)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None)
    initialization.add_argument("--finetune-from", default=None)
    parser.add_argument("--rule-opponent-weight", type=float, default=1.0)
    parser.add_argument("--champion-opponent-weight", type=float, default=0.0)
    parser.add_argument("--history-opponent-weight", type=float, default=0.0)
    parser.add_argument("--champion-checkpoint", default=None)
    parser.add_argument("--history-checkpoints", nargs="*", default=[])

    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument(
        "--benchmark-workers", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument("--benchmark-warmup-deals", type=int, default=100)
    parser.add_argument("--benchmark-deals", type=int, default=2000)
    parser.add_argument("--benchmark-sustained-seconds", type=float, default=600.0)
    parser.add_argument("--benchmark-json", default=None)
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
    opponent_pool = _opponent_pool_config(args)
    active_opponent_checkpoints: list[str] = []
    if opponent_pool.champion_weight > 0.0:
        assert opponent_pool.champion_checkpoint is not None
        active_opponent_checkpoints.append(opponent_pool.champion_checkpoint)
    if opponent_pool.history_weight > 0.0:
        active_opponent_checkpoints.extend(opponent_pool.history_checkpoints)
    for checkpoint in active_opponent_checkpoints:
        actor_path = Path(checkpoint)
        metadata_path = actor_path.with_name(f"{actor_path.name}.json")
        if not actor_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"opponent actor and metadata sidecar are required: {actor_path}"
            )
    if args.finetune_from and not Path(args.finetune_from).is_file():
        raise FileNotFoundError(f"fine-tune trainer checkpoint not found: {args.finetune_from}")
    if args.benchmark_only:
        if any(type(value) is not int or value <= 0 for value in args.benchmark_workers):
            raise ValueError("--benchmark-workers must contain positive integers")
        if args.benchmark_warmup_deals < 0 or args.benchmark_deals <= 0:
            raise ValueError("benchmark deal counts are invalid")
        if not math.isfinite(args.benchmark_sustained_seconds) or args.benchmark_sustained_seconds < 0:
            raise ValueError("--benchmark-sustained-seconds must be finite and nonnegative")


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
        *legacy_defaults,
    ):
        saved_value = saved_run.get(key, legacy_defaults.get(key))
        if saved_value != getattr(args, key):
            raise ValueError(
                f"resume checkpoint {key!r} differs from CLI; exact continuation refused"
            )


def _solver_latencies(batch: CollectionBatch) -> np.ndarray:
    return np.asarray(
        [
            room.solver_seconds
            for deal in batch.deals
            for room in (deal.room_team0, deal.room_team1)
        ],
        dtype=np.float64,
    )


def _collection_metrics(batch: CollectionBatch) -> dict[str, Any]:
    latencies = _solver_latencies(batch)
    margins = [deal.duplicate_margin_points for deal in batch.deals]
    opponent_counts = Counter(deal.opponent_id for deal in batch.deals)
    return {
        "duplicate_deals": len(batch.deals),
        "games": len(batch.deals) * 2,
        "solver_calls": batch.solver_calls,
        "scanned_candidates": batch.scanned_candidates,
        "nil_filtered_candidates": batch.nil_filtered_candidates,
        "nil_filter_rate": (
            batch.nil_filtered_candidates / batch.scanned_candidates
            if batch.scanned_candidates
            else 0.0
        ),
        "elapsed_seconds": batch.elapsed_seconds,
        "accepted_games_per_second": len(batch.deals) * 2 / batch.elapsed_seconds,
        "mean_duplicate_margin_points": statistics.fmean(margins),
        "opponent_deal_counts": dict(sorted(opponent_counts.items())),
        "opponent_deal_fractions": {
            key: count / len(batch.deals)
            for key, count in sorted(opponent_counts.items())
        },
        "solver_seconds_p50": float(np.percentile(latencies, 50)),
        "solver_seconds_p95": float(np.percentile(latencies, 95)),
        "solver_seconds_p99": float(np.percentile(latencies, 99)),
        "solver_seconds_max": float(latencies.max()),
        "worker_peak_rss_bytes": batch.worker_peak_rss_bytes,
        "aggregate_worker_peak_rss_bytes": batch.aggregate_worker_peak_rss_bytes,
    }


def _atomic_write_json(destination: Path, payload: Any) -> None:
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


def _run_sustained_benchmark(
    actor: PolicyMLP,
    args: argparse.Namespace,
    *,
    workers: int,
    start_candidate_index: int,
) -> dict[str, Any] | None:
    if args.benchmark_sustained_seconds <= 0:
        return None
    target_seconds = args.benchmark_sustained_seconds
    batch_summaries: list[tuple[int, float, int, int]] = []
    cursor = start_candidate_index
    wall_started = time.perf_counter()
    with ProductionDuplicateCollector(
        workers=workers,
        actor_hidden_dims=args.actor_hidden_dims,
        bid_policy_seed=args.bid_policy_seed,
        opponent_pool_config=_opponent_pool_config(args),
    ) as collector:
        while time.perf_counter() - wall_started < target_seconds:
            batch = collector.collect(
                actor,
                start_candidate_index=cursor,
                target_deals=min(256, args.benchmark_deals),
                base_shuffle_seed=args.base_shuffle_seed,
                run_seed=args.seed,
                deterministic=False,
                record_transitions=True,
            )
            batch_summaries.append(
                (
                    len(batch.deals),
                    batch.elapsed_seconds,
                    batch.worker_peak_rss_bytes,
                    batch.aggregate_worker_peak_rss_bytes,
                )
            )
            cursor = batch.next_candidate_index
    wall_elapsed = time.perf_counter() - wall_started
    deals = sum(item[0] for item in batch_summaries)
    first_window = batch_summaries[0][1] if batch_summaries else 0.0
    first_rate = batch_summaries[0][0] * 2 / first_window if first_window else 0.0
    last_batches = batch_summaries[-max(1, len(batch_summaries) // 2) :]
    last_deals = sum(item[0] for item in last_batches)
    last_seconds = sum(item[1] for item in last_batches)
    return {
        "workers": workers,
        "wall_seconds": wall_elapsed,
        "duplicate_deals": deals,
        "games": deals * 2,
        "games_per_second": deals * 2 / wall_elapsed,
        "first_batch_games_per_second": first_rate,
        "last_half_games_per_second": last_deals * 2 / last_seconds,
        "next_candidate_index": cursor,
        "peak_worker_rss_bytes": max(item[2] for item in batch_summaries),
        "peak_aggregate_worker_rss_bytes": max(item[3] for item in batch_summaries),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _set_seeds(args.seed)
    actor = PolicyMLP(
        input_dim=FirstFourFeatureEncoderV2.TOTAL_DIM,
        hidden_dims=args.actor_hidden_dims,
        output_dim=52,
    ).cpu()
    worker_results: list[dict[str, Any]] = []
    next_cursors: dict[int, int] = {}
    best_batch: CollectionBatch | None = None
    best_measured_rate = -math.inf
    for workers in args.benchmark_workers:
        cursor = 0
        with ProductionDuplicateCollector(
            workers=workers,
            actor_hidden_dims=args.actor_hidden_dims,
            bid_policy_seed=args.bid_policy_seed,
            opponent_pool_config=_opponent_pool_config(args),
        ) as collector:
            if args.benchmark_warmup_deals:
                warmup = collector.collect(
                    actor,
                    start_candidate_index=cursor,
                    target_deals=args.benchmark_warmup_deals,
                    base_shuffle_seed=args.base_shuffle_seed,
                    run_seed=args.seed,
                    deterministic=False,
                    record_transitions=True,
                )
                cursor = warmup.next_candidate_index
            measured = collector.collect(
                actor,
                start_candidate_index=cursor,
                target_deals=args.benchmark_deals,
                base_shuffle_seed=args.base_shuffle_seed,
                run_seed=args.seed,
                deterministic=False,
                record_transitions=True,
            )
        metrics = _collection_metrics(measured)
        metrics["workers"] = workers
        worker_results.append(metrics)
        if metrics["accepted_games_per_second"] > best_measured_rate:
            best_measured_rate = float(metrics["accepted_games_per_second"])
            best_batch = measured
        next_cursors[workers] = measured.next_candidate_index
        print(json.dumps({"benchmark": metrics}, ensure_ascii=False), flush=True)

    baseline = min(worker_results, key=lambda item: item["workers"])
    baseline_rate = float(baseline["accepted_games_per_second"])
    for metrics in worker_results:
        metrics["parallel_speedup_vs_baseline"] = (
            float(metrics["accepted_games_per_second"]) / baseline_rate
        )
    best = max(worker_results, key=lambda item: item["accepted_games_per_second"])
    best_workers = int(best["workers"])
    if best_batch is None:
        raise AssertionError("benchmark did not retain a measured trajectory batch")
    source_transitions = best_batch.transitions
    target_transition_count = args.rollout_deals * 16
    transition_repeats = int(
        math.ceil(target_transition_count / len(source_transitions))
    )
    benchmark_transitions = (source_transitions * transition_repeats)[
        :target_transition_count
    ]
    benchmark_actor = PolicyMLP(
        input_dim=FirstFourFeatureEncoderV2.TOTAL_DIM,
        hidden_dims=args.actor_hidden_dims,
        output_dim=52,
    ).cpu()
    benchmark_actor.load_state_dict(actor.state_dict())
    benchmark_critic = ValueMLP(536, args.critic_hidden_dims).cpu()
    benchmark_ppo_config = _ppo_config(args)
    benchmark_optimizer = build_optimizer(
        benchmark_actor, benchmark_critic, benchmark_ppo_config
    )
    ppo_started = time.perf_counter()
    ppo_stats = ppo_update(
        benchmark_actor,
        benchmark_critic,
        benchmark_optimizer,
        benchmark_transitions,
        benchmark_ppo_config,
        device="cpu",
        shuffle_seed=args.seed,
    )
    ppo_update_seconds = time.perf_counter() - ppo_started
    best_batch = None
    benchmark_transitions = ()
    sustained = _run_sustained_benchmark(
        actor,
        args,
        workers=best_workers,
        start_candidate_index=next_cursors[best_workers],
    )
    stable_rate = (
        float(sustained["last_half_games_per_second"])
        if sustained is not None
        else float(best["accepted_games_per_second"])
    )
    projections: dict[str, dict[str, float | int]] = {}
    for games in (200_000, 500_000, 1_000_000):
        duplicate_deals = games // 2
        updates = int(math.ceil(duplicate_deals / args.rollout_deals))
        collection_seconds = games / stable_rate
        total_seconds = collection_seconds + updates * ppo_update_seconds
        projections[str(games)] = {
            "duplicate_deals": duplicate_deals,
            "updates": updates,
            "collection_seconds": collection_seconds,
            "ppo_seconds": updates * ppo_update_seconds,
            "total_seconds": total_seconds,
            "total_hours": total_seconds / 3600.0,
        }
    recommendation = (
        "local"
        if float(projections["500000"]["total_hours"]) <= 12.0
        else "multi-core-cpu-cloud"
    )
    report = {
        "schema": "solver-leaf-local-benchmark-v1",
        "encoder_schema": ENCODER_SCHEMA,
        "input_dim": 536,
        "output_dim": 52,
        "worker_results": worker_results,
        "speedup_baseline_workers": int(baseline["workers"]),
        "best_workers": best_workers,
        "sustained": sustained,
        "stable_games_per_second": stable_rate,
        "ppo_update_seconds": ppo_update_seconds,
        "ppo_benchmark_rollout_deals": args.rollout_deals,
        "ppo_update": asdict(ppo_stats),
        "projections": projections,
        "recommendation": recommendation,
        "gpu_recommended": False,
    }
    print(json.dumps({"benchmark_summary": report}, ensure_ascii=False), flush=True)
    if args.benchmark_json:
        _atomic_write_json(Path(args.benchmark_json), report)
    return report


def _save_artifacts(
    args: argparse.Namespace,
    actor: PolicyMLP,
    critic: ValueMLP,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    *,
    update: int,
    deals_trained: int,
    candidate_cursor: int,
    suffix: str,
    run_config: dict[str, Any] | None = None,
) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_config = vars(args).copy() if run_config is None else dict(run_config)
    save_training_checkpoint(
        save_dir / f"trainer_{suffix}.pt",
        actor,
        critic,
        optimizer,
        update=update,
        deals_trained=deals_trained,
        candidate_cursor=candidate_cursor,
        best_validation_margin=None,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        ppo_config=ppo_config,
        run_config=resolved_run_config,
    )
    export_actor(
        save_dir / f"actor_{suffix}.pt",
        actor,
        actor_hidden_dims=args.actor_hidden_dims,
        training_update=update,
        deals_trained=deals_trained,
        residual_checkpoint_sha256=DEPLOYED_CHECKPOINT_SHA256,
        extra_metadata={
            "ppo_config": asdict(ppo_config),
            "training_config": resolved_run_config,
        },
    )


def train(args: argparse.Namespace) -> None:
    _set_seeds(args.seed)
    device = torch.device(args.device)
    actor = PolicyMLP(
        input_dim=FirstFourFeatureEncoderV2.TOTAL_DIM,
        hidden_dims=args.actor_hidden_dims,
        output_dim=52,
    ).to(device)
    critic = ValueMLP(536, args.critic_hidden_dims).to(device)
    ppo_config = _ppo_config(args)
    update = 0
    deals_trained = 0
    candidate_cursor = 0
    artifact_run_config = vars(args).copy()
    finetune_source: dict[str, Any] | None = None
    if args.finetune_from:
        source = load_finetune_weights(
            Path(args.finetune_from), actor, critic, map_location=device
        )
        if source.config["actor_hidden_dims"] != list(args.actor_hidden_dims):
            raise ValueError("fine-tune checkpoint actor architecture differs from CLI")
        if source.config["critic_hidden_dims"] != list(args.critic_hidden_dims):
            raise ValueError("fine-tune checkpoint critic architecture differs from CLI")
        finetune_source = {
            "path": str(Path(args.finetune_from)),
            "update": source.update,
            "deals_trained": source.deals_trained,
            "candidate_cursor": source.candidate_cursor,
        }
    optimizer = build_optimizer(actor, critic, ppo_config)
    if args.resume:
        resume = load_training_checkpoint(
            Path(args.resume), actor, critic, optimizer, map_location=device
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
                    "encoder_schema": ENCODER_SCHEMA,
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

    with ProductionDuplicateCollector(
        workers=args.workers,
        actor_hidden_dims=args.actor_hidden_dims,
        bid_policy_seed=args.bid_policy_seed,
        opponent_pool_config=_opponent_pool_config(args),
    ) as collector:
        while deals_trained < target_deals:
            rollout_deals = min(args.rollout_deals, target_deals - deals_trained)
            collection = collector.collect(
                actor,
                start_candidate_index=candidate_cursor,
                target_deals=rollout_deals,
                base_shuffle_seed=args.base_shuffle_seed,
                run_seed=args.seed,
                deterministic=False,
                record_transitions=True,
            )
            candidate_cursor = collection.next_candidate_index
            update_started = time.perf_counter()
            stats = ppo_update(
                actor,
                critic,
                optimizer,
                collection.transitions,
                ppo_config,
                device=device,
                shuffle_seed=args.seed + update,
            )
            update_seconds = time.perf_counter() - update_started
            deals_trained += len(collection.deals)
            update += 1
            metrics = _collection_metrics(collection)
            metrics.update(
                {
                    "update": update,
                    "deals_trained": deals_trained,
                    "games_trained": deals_trained * 2,
                    "candidate_cursor": candidate_cursor,
                    "ppo_seconds": update_seconds,
                    "ppo": asdict(stats),
                }
            )
            print(json.dumps({"training_update": metrics}, ensure_ascii=False), flush=True)
            if update % args.save_every_updates == 0:
                _save_artifacts(
                    args,
                    actor,
                    critic,
                    optimizer,
                    ppo_config,
                    update=update,
                    deals_trained=deals_trained,
                    candidate_cursor=candidate_cursor,
                    suffix=f"update_{update:06d}",
                    run_config=artifact_run_config,
                )

    _save_artifacts(
        args,
        actor,
        critic,
        optimizer,
        ppo_config,
        update=update,
        deals_trained=deals_trained,
        candidate_cursor=candidate_cursor,
        suffix="final",
        run_config=artifact_run_config,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    if args.benchmark_only:
        benchmark(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
