"""Paired evaluation of one exported solver-leaf PPO actor versus the rule player."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from residual_bidder.deployment import DEPLOYED_CHECKPOINT_SHA256
from rl.solver_leaf_env import ProductionDuplicateCollector
from rl.solver_leaf_ppo import load_exported_actor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one 536-d PPO actor against RuleBasedFirst4Player",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=536_500)
    parser.add_argument("--base-shuffle-seed", type=int, default=153_600_000)
    parser.add_argument("--bid-policy-seed", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("deals", "workers"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("seed", "base_shuffle_seed"):
        value = getattr(args, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")


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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    actor, metadata = load_exported_actor(Path(args.checkpoint), device="cpu")
    if metadata.get("residual_checkpoint_sha256") != DEPLOYED_CHECKPOINT_SHA256:
        raise ValueError(
            "actor was trained with a different production Residual checkpoint"
        )
    hidden_dims = metadata["hidden_dims"]
    with ProductionDuplicateCollector(
        workers=args.workers,
        actor_hidden_dims=hidden_dims,
        bid_policy_seed=args.bid_policy_seed,
    ) as collector:
        batch = collector.collect(
            actor,
            start_candidate_index=0,
            target_deals=args.deals,
            base_shuffle_seed=args.base_shuffle_seed,
            run_seed=args.seed,
            deterministic=True,
            record_transitions=False,
        )

    margins = [deal.duplicate_margin_points for deal in batch.deals]
    room_team0 = [deal.room_team0.candidate_margin_points for deal in batch.deals]
    room_team1 = [deal.room_team1.candidate_margin_points for deal in batch.deals]
    mean = statistics.fmean(margins)
    standard_error = (
        statistics.stdev(margins) / math.sqrt(len(margins))
        if len(margins) > 1
        else 0.0
    )
    ci_half_width = 1.96 * standard_error
    solver_seconds = np.asarray(
        [
            room.solver_seconds
            for deal in batch.deals
            for room in (deal.room_team0, deal.room_team1)
        ],
        dtype=np.float64,
    )
    report: dict[str, Any] = {
        "schema": "solver-leaf-ppo-vs-rule-evaluation-v1",
        "comparison": "new-ppo-vs-RuleBasedFirst4Player",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "actor_metadata": metadata,
        "duplicate_deals": len(batch.deals),
        "games": len(batch.deals) * 2,
        "solver_calls": batch.solver_calls,
        "mean_duplicate_margin_points": mean,
        "standard_error_points": standard_error,
        "confidence_interval_95_points": [mean - ci_half_width, mean + ci_half_width],
        "success_lower_ci_above_zero": mean - ci_half_width > 0.0,
        "wins": sum(value > 0.0 for value in margins),
        "ties": sum(value == 0.0 for value in margins),
        "losses": sum(value < 0.0 for value in margins),
        "mean_room_candidate_team0_margin_points": statistics.fmean(room_team0),
        "mean_room_candidate_team1_margin_points": statistics.fmean(room_team1),
        "scanned_candidates": batch.scanned_candidates,
        "nil_filtered_candidates": batch.nil_filtered_candidates,
        "nil_filter_rate": batch.nil_filtered_candidates / batch.scanned_candidates,
        "elapsed_seconds": batch.elapsed_seconds,
        "accepted_games_per_second": len(batch.deals) * 2 / batch.elapsed_seconds,
        "solver_seconds_p50": float(np.percentile(solver_seconds, 50)),
        "solver_seconds_p95": float(np.percentile(solver_seconds, 95)),
        "solver_seconds_p99": float(np.percentile(solver_seconds, 99)),
        "worker_peak_rss_bytes": batch.worker_peak_rss_bytes,
        "aggregate_worker_peak_rss_bytes": batch.aggregate_worker_peak_rss_bytes,
        "workers": args.workers,
        "base_shuffle_seed": args.base_shuffle_seed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if args.output_json:
        _atomic_write_json(Path(args.output_json), report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    evaluate(args)


if __name__ == "__main__":
    main()
