"""Compare one residual bidder with frozen NSFP in fast paired duplicate deals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from residual_bidder.checkpoint import (
    CalibrationTuple,
    load_checkpoint,
    promote_meta,
)
from residual_bidder.config import BidderConfig, ConfigError, canonical_sha256
from residual_bidder.evaluation import evaluate_fast_duplicates
from residual_bidder.nsfp import FrozenNSFP
from residual_bidder.policy import NSFPArgmaxPolicy, StochasticResidualPolicy
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _play_pipeline_sha256(config: BidderConfig) -> str:
    return canonical_sha256(
        {
            "play_config_sha256": config.play.config_sha256,
            "source_manifest": [
                [source, _sha256(Path(source))]
                for source in config.play.source_manifest
            ],
        }
    )


def _checkpoint_dataset_sha256(checkpoint: Path) -> str:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
        raise ValueError("experimental checkpoint has no safe metadata mapping")
    value = artifact["metadata"].get("dataset_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("experimental checkpoint has no dataset SHA-256")
    return value


def evaluate_checkpoint(
    config: BidderConfig,
    *,
    checkpoint: Path | None,
    start_seed: int,
    deals: int,
    calibration: CalibrationTuple,
    policy_seed: int,
    control_nsfp: bool = False,
) -> dict[str, object]:
    if type(start_seed) is not int or start_seed < 0 or type(deals) is not int or deals <= 0:
        raise ValueError("start_seed must be nonnegative and deals must be positive")
    torch.set_num_threads(1)
    nsfp = FrozenNSFP.load(Path(config.nsfp.path), config.nsfp.sha256, torch.device("cpu"))
    opponent = NSFPArgmaxPolicy(nsfp)
    model_id = "legacy-nsfp-control"
    policy_id = opponent.policy_id
    if control_nsfp:
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
        promoted = promote_meta(candidate_meta, calibration)
        candidate = StochasticResidualPolicy(nsfp, ensemble, promoted)
        model_id = candidate_meta.model_id
        policy_id = candidate.policy_id

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
        "schema": "minimal-fast-duplicate-evaluation-v1",
        "control_nsfp": control_nsfp,
        "model_id": model_id,
        "policy_id": policy_id,
        "calibration": {
            "uncertainty_lambda": calibration.uncertainty_lambda,
            "temperature": calibration.temperature,
            "epsilon": calibration.epsilon,
            "rho": calibration.rho,
        },
        "start_seed": start_seed,
        "deals": summary.deals,
        "solver_calls": summary.solver_calls,
        "wall_seconds": wall_seconds,
        "deals_per_hour": summary.deals / wall_seconds * 3600.0,
        "mean_duplicate_margin": summary.mean_duplicate_margin,
        "standard_error": summary.standard_error,
        "wins": summary.wins,
        "ties": summary.ties,
        "losses": summary.losses,
        "deal_margins": [result.duplicate_margin for result in summary.results],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_bidder/base.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--control-nsfp", action="store_true")
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--deals", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=20260721)
    parser.add_argument("--uncertainty-lambda", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = evaluate_checkpoint(
            BidderConfig.load(arguments.config),
            checkpoint=arguments.checkpoint,
            start_seed=arguments.start_seed,
            deals=arguments.deals,
            calibration=CalibrationTuple(
                arguments.uncertainty_lambda,
                arguments.temperature,
                arguments.epsilon,
                arguments.rho,
            ),
            policy_seed=arguments.policy_seed,
            control_nsfp=arguments.control_nsfp,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
