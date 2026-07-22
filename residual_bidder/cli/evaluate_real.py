"""Evaluate two acting bidders with the unchanged full-play player."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from residual_bidder.checkpoint import CalibrationTuple, load_checkpoint, promote_meta
from residual_bidder.cli.evaluate_fast import (
    _checkpoint_dataset_sha256,
    _play_pipeline_sha256,
)
from residual_bidder.config import BidderConfig, ConfigError
from residual_bidder.nsfp import FrozenNSFP
from residual_bidder.policy import NSFPArgmaxPolicy, StochasticResidualPolicy
from residual_bidder.real_evaluation import evaluate_real_duplicates
from strategy.hyperparam_config import HyperparamConfig
from strategy.rule_exact_first4_nil_player import RuleExactFirst4NilPlayer
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


def evaluate_checkpoint_real(
    config: BidderConfig,
    *,
    checkpoint: Path | None,
    opponent_checkpoint: Path | None,
    start_seed: int,
    deals: int,
    policy_seed: int,
    inner_workers: int,
    control_nsfp: bool,
) -> dict[str, object]:
    if type(start_seed) is not int or start_seed < 0:
        raise ValueError("start_seed must be a nonnegative integer")
    if type(deals) is not int or deals <= 0:
        raise ValueError("deals must be a positive integer")
    if type(inner_workers) is not int or inner_workers <= 0:
        raise ValueError("inner_workers must be a positive integer")

    torch.set_num_threads(1)
    nsfp = FrozenNSFP.load(Path(config.nsfp.path), config.nsfp.sha256, torch.device("cpu"))
    opponent = NSFPArgmaxPolicy(nsfp)
    opponent_model_id = "legacy-nsfp"
    opponent_policy_id = opponent.policy_id
    model_id = "legacy-nsfp-control"
    policy_id = opponent.policy_id
    if control_nsfp:
        if opponent_checkpoint is not None:
            raise ValueError("--opponent-checkpoint cannot be used with --control-nsfp")
        candidate = opponent
    else:
        if checkpoint is None:
            raise ValueError("--checkpoint is required outside NSFP control mode")
        ensemble, candidate_meta = load_checkpoint(
            checkpoint,
            expected_nsfp_sha256=config.nsfp.sha256,
            expected_play_pipeline_sha256=_play_pipeline_sha256(config),
            expected_config_sha256=config.sha256(),
            expected_dataset_manifest_sha256=_checkpoint_dataset_sha256(checkpoint),
        )
        calibration = CalibrationTuple(0.0, 0.0, 0.0, 1.0)
        candidate = StochasticResidualPolicy(
            nsfp, ensemble, promote_meta(candidate_meta, calibration)
        )
        model_id = candidate_meta.model_id
        policy_id = candidate.policy_id

    if opponent_checkpoint is not None:
        opponent_ensemble, opponent_meta = load_checkpoint(
            opponent_checkpoint,
            expected_nsfp_sha256=config.nsfp.sha256,
            expected_play_pipeline_sha256=_play_pipeline_sha256(config),
            expected_config_sha256=config.sha256(),
            expected_dataset_manifest_sha256=_checkpoint_dataset_sha256(
                opponent_checkpoint
            ),
        )
        opponent_calibration = CalibrationTuple(0.0, 0.0, 0.0, 1.0)
        opponent = StochasticResidualPolicy(
            nsfp,
            opponent_ensemble,
            promote_meta(opponent_meta, opponent_calibration),
        )
        opponent_model_id = opponent_meta.model_id
        opponent_policy_id = opponent.policy_id

    solver = ExactDoubleDummyCppFastestSolver()
    if not solver.native_available:
        raise RuntimeError("native terminal solver is unavailable")
    hyperparameters = HyperparamConfig.from_yaml(config.play.config_path)

    def player_factory() -> RuleExactFirst4NilPlayer:
        return RuleExactFirst4NilPlayer(
            exact_solver=solver,
            exact_threshold=config.play.exact_threshold,
            bid_model=None,
            bid_device="cpu",
            hyperparam_config=hyperparameters,
            num_workers=inner_workers,
        )

    started = time.perf_counter()
    summary = evaluate_real_duplicates(
        [start_seed + index for index in range(deals)],
        candidate,
        opponent,
        player_factory,
        policy_seed=policy_seed,
    )
    wall_seconds = time.perf_counter() - started
    return {
        "ok": True,
        "schema": "minimal-real-duplicate-evaluation-v1",
        "control_nsfp": control_nsfp,
        "model_id": model_id,
        "policy_id": policy_id,
        "opponent_model_id": opponent_model_id,
        "opponent_policy_id": opponent_policy_id,
        "calibration": {
            "uncertainty_lambda": 0.0,
            "temperature": 0.0,
            "epsilon": 0.0,
            "rho": 1.0,
        },
        "play_player": "RuleExactFirst4NilPlayer",
        "play_config": config.play.config_path,
        "belief_bidder": config.nsfp.path,
        "start_seed": start_seed,
        "deals": summary.deals,
        "inner_workers": inner_workers,
        "wall_seconds": wall_seconds,
        "deals_per_hour": summary.deals / wall_seconds * 3600.0,
        "mean_duplicate_margin": summary.mean_duplicate_margin,
        "standard_error": summary.standard_error,
        "wins": summary.wins,
        "ties": summary.ties,
        "losses": summary.losses,
        "deal_margins": [result.duplicate_margin for result in summary.results],
        "rooms": [
            {
                "shuffle_seed": result.shuffle_seed,
                "candidate_team0_margin": result.room_team0_margin,
                "candidate_team1_margin": result.room_team1_margin,
            }
            for result in summary.results
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_bidder/base.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--control-nsfp", action="store_true")
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--deals", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=20260721)
    parser.add_argument("--inner-workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = evaluate_checkpoint_real(
            BidderConfig.load(arguments.config),
            checkpoint=arguments.checkpoint,
            opponent_checkpoint=arguments.opponent_checkpoint,
            start_seed=arguments.start_seed,
            deals=arguments.deals,
            policy_seed=arguments.policy_seed,
            inner_workers=arguments.inner_workers,
            control_nsfp=arguments.control_nsfp,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
