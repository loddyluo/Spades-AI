"""Compare two calibrations of one residual bidder in fast duplicate play."""

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
from residual_bidder.evaluation import evaluate_fast_duplicates
from residual_bidder.nsfp import FrozenNSFP
from residual_bidder.policy import StochasticResidualPolicy
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


def evaluate_calibration_pair(
    config: BidderConfig,
    *,
    checkpoint: Path,
    start_seed: int,
    deals: int,
    candidate_calibration: CalibrationTuple,
    opponent_calibration: CalibrationTuple,
    policy_seed: int,
) -> dict[str, object]:
    if type(start_seed) is not int or start_seed < 0:
        raise ValueError("start_seed must be a nonnegative integer")
    if type(deals) is not int or deals <= 0:
        raise ValueError("deals must be a positive integer")

    torch.set_num_threads(1)
    nsfp = FrozenNSFP.load(
        Path(config.nsfp.path), config.nsfp.sha256, torch.device("cpu")
    )
    ensemble, candidate_meta = load_checkpoint(
        checkpoint,
        expected_nsfp_sha256=config.nsfp.sha256,
        expected_play_pipeline_sha256=_play_pipeline_sha256(config),
        expected_config_sha256=config.sha256(),
        expected_dataset_manifest_sha256=_checkpoint_dataset_sha256(checkpoint),
    )
    candidate = StochasticResidualPolicy(
        nsfp, ensemble, promote_meta(candidate_meta, candidate_calibration)
    )
    opponent = StochasticResidualPolicy(
        nsfp, ensemble, promote_meta(candidate_meta, opponent_calibration)
    )

    solver = ExactDoubleDummyCppFastestSolver()
    if not solver.native_available:
        raise RuntimeError("native terminal solver is unavailable")
    started = time.perf_counter()
    summary = evaluate_fast_duplicates(
        [start_seed + index for index in range(deals)],
        candidate,
        opponent,
        solver,
        policy_seed=policy_seed,
    )
    wall_seconds = time.perf_counter() - started
    return {
        "ok": True,
        "schema": "minimal-fast-calibration-pair-v1",
        "model_id": candidate_meta.model_id,
        "candidate_policy_id": candidate.policy_id,
        "opponent_policy_id": opponent.policy_id,
        "candidate_calibration": {
            "uncertainty_lambda": candidate_calibration.uncertainty_lambda,
            "temperature": candidate_calibration.temperature,
            "epsilon": candidate_calibration.epsilon,
            "rho": candidate_calibration.rho,
        },
        "opponent_calibration": {
            "uncertainty_lambda": opponent_calibration.uncertainty_lambda,
            "temperature": opponent_calibration.temperature,
            "epsilon": opponent_calibration.epsilon,
            "rho": opponent_calibration.rho,
        },
        "start_seed": start_seed,
        "deals": summary.deals,
        "wall_seconds": wall_seconds,
        "deals_per_hour": summary.deals / wall_seconds * 3600.0,
        "mean_duplicate_margin": summary.mean_duplicate_margin,
        "standard_error": summary.standard_error,
        "wins": summary.wins,
        "ties": summary.ties,
        "losses": summary.losses,
        "deal_margins": [result.duplicate_margin for result in summary.results],
    }


def _calibration(arguments: argparse.Namespace, prefix: str) -> CalibrationTuple:
    return CalibrationTuple(
        getattr(arguments, f"{prefix}_lambda"),
        getattr(arguments, f"{prefix}_temperature"),
        getattr(arguments, f"{prefix}_epsilon"),
        getattr(arguments, f"{prefix}_rho"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_bidder/base.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--deals", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=20260721)
    for prefix in ("candidate", "opponent"):
        parser.add_argument(f"--{prefix}-lambda", type=float, required=True)
        parser.add_argument(f"--{prefix}-temperature", type=float, default=0.0)
        parser.add_argument(f"--{prefix}-epsilon", type=float, default=0.0)
        parser.add_argument(f"--{prefix}-rho", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = evaluate_calibration_pair(
            BidderConfig.load(arguments.config),
            checkpoint=arguments.checkpoint,
            start_seed=arguments.start_seed,
            deals=arguments.deals,
            candidate_calibration=_calibration(arguments, "candidate"),
            opponent_calibration=_calibration(arguments, "opponent"),
            policy_seed=arguments.policy_seed,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
