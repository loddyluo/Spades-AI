"""Paired evaluation of a four-role Nil actor bundle."""

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
from rl.nil_solver_leaf_env import NilProductionDuplicateCollector
from rl.nil_solver_leaf_ppo import load_nil_role_actor_bundle
from rl.solver_leaf_env import OpponentPoolConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate four role-specific Nil PPO actors as one bundle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--opponent-bundle", default=None)
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=636_500)
    parser.add_argument("--base-shuffle-seed", type=int, default=163_600_000)
    parser.add_argument("--bid-policy-seed", type=int, default=None)
    parser.add_argument("--oversample-factor", type=float, default=6.5)
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
    if not math.isfinite(args.oversample_factor) or args.oversample_factor < 1.0:
        raise ValueError("--oversample-factor must be finite and at least one")


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
    candidate_path = Path(args.bundle).resolve()
    actors, manifest, metadata = load_nil_role_actor_bundle(candidate_path, device="cpu")
    hidden_dims = metadata[manifest["roles"][0]]["hidden_dims"]
    for role, sidecar in metadata.items():
        if sidecar.get("residual_checkpoint_sha256") != DEPLOYED_CHECKPOINT_SHA256:
            raise ValueError(f"candidate role {role} used a different Residual bidder")

    opponent_path: Path | None = None
    opponent_manifest: dict[str, Any] | None = None
    if args.opponent_bundle:
        opponent_path = Path(args.opponent_bundle).resolve()
        _, opponent_manifest, opponent_metadata = load_nil_role_actor_bundle(
            opponent_path, device="cpu"
        )
        for role, sidecar in opponent_metadata.items():
            if sidecar.get("residual_checkpoint_sha256") != DEPLOYED_CHECKPOINT_SHA256:
                raise ValueError(f"opponent role {role} used a different Residual bidder")
            if sidecar["hidden_dims"] != hidden_dims:
                raise ValueError("candidate and opponent bundle architectures differ")
        opponent_pool = OpponentPoolConfig(
            rule_weight=0.0,
            champion_weight=1.0,
            champion_checkpoint=str(opponent_path),
        )
    else:
        opponent_pool = OpponentPoolConfig()

    with NilProductionDuplicateCollector(
        workers=args.workers,
        actor_hidden_dims=hidden_dims,
        bid_policy_seed=args.bid_policy_seed,
        opponent_pool_config=opponent_pool,
        oversample_factor=args.oversample_factor,
    ) as collector:
        batch = collector.collect(
            actors,
            start_candidate_index=0,
            target_deals=args.deals,
            base_shuffle_seed=args.base_shuffle_seed,
            run_seed=args.seed,
            deterministic=True,
            record_transitions=False,
        )

    margins = [deal.duplicate_margin_points for deal in batch.deals]
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
        "schema": "solver-leaf-nil-four-role-evaluation-v1",
        "comparison": (
            "candidate-nil-bundle-vs-frozen-nil-bundle"
            if opponent_path is not None
            else "candidate-nil-bundle-vs-RuleBasedFirst4NilPlayer"
        ),
        "bundle": str(candidate_path),
        "opponent_bundle": None if opponent_path is None else str(opponent_path),
        "bundle_manifest": manifest,
        "opponent_bundle_manifest": opponent_manifest,
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
        "mean_room_candidate_team0_margin_points": statistics.fmean(
            deal.room_team0.candidate_margin_points for deal in batch.deals
        ),
        "mean_room_candidate_team1_margin_points": statistics.fmean(
            deal.room_team1.candidate_margin_points for deal in batch.deals
        ),
        "scanned_candidates": batch.scanned_candidates,
        "nil_count_histogram": {
            str(key): value for key, value in batch.nil_count_histogram.items()
        },
        "exactly_one_nil_rate": len(batch.deals) / batch.scanned_candidates,
        "elapsed_seconds": batch.elapsed_seconds,
        "accepted_games_per_second": len(batch.deals) * 2 / batch.elapsed_seconds,
        "solver_seconds_p50": float(np.percentile(solver_seconds, 50)),
        "solver_seconds_p95": float(np.percentile(solver_seconds, 95)),
        "solver_seconds_p99": float(np.percentile(solver_seconds, 99)),
        "worker_peak_rss_bytes": batch.worker_peak_rss_bytes,
        "aggregate_worker_peak_rss_bytes": batch.aggregate_worker_peak_rss_bytes,
        "workers": args.workers,
        "seed": args.seed,
        "base_shuffle_seed": args.base_shuffle_seed,
        "bid_policy_seed": args.bid_policy_seed,
        "oversample_factor": args.oversample_factor,
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
