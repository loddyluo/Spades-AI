"""Generate the minimal four-trick-plus-DDS residual-bidder dataset."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from residual_bidder.config import BidderConfig, ConfigError
from residual_bidder.hybrid import generate_hybrid_deal, save_hybrid_npz
from residual_bidder.nsfp import FrozenNSFP
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


_WORKER_NSFP: FrozenNSFP | None = None
_WORKER_SOLVER: ExactDoubleDummyCppFastestSolver | None = None


def _initialize_worker(checkpoint: str, checkpoint_sha256: str) -> None:
    global _WORKER_NSFP, _WORKER_SOLVER
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_NSFP = FrozenNSFP.load(
        Path(checkpoint), checkpoint_sha256, torch.device("cpu")
    )
    _WORKER_SOLVER = ExactDoubleDummyCppFastestSolver()
    if not _WORKER_SOLVER.native_available:
        raise RuntimeError("native terminal solver is unavailable")


def _generate_worker(shuffle_seed: int):
    if _WORKER_NSFP is None or _WORKER_SOLVER is None:
        raise RuntimeError("hybrid worker was not initialized")
    return generate_hybrid_deal(shuffle_seed, _WORKER_NSFP, _WORKER_SOLVER)


def _validate_minimal_config(config: BidderConfig) -> None:
    if config.play.first_tricks != 4:
        raise ValueError("minimal hybrid generation requires exactly four first tricks")
    if not config.play.enable_nil or config.play.enable_blind_nil:
        raise ValueError("minimal hybrid generation requires Nil on and Blind Nil off")
    if config.targets.divisor != 100.0:
        raise ValueError("minimal hybrid generation requires target divisor 100")


def generate_to_npz(
    config: BidderConfig,
    *,
    start_seed: int,
    deals: int,
    output: Path,
    workers: int = 1,
    nsfp: Any | None = None,
    solver: Any | None = None,
) -> dict[str, object]:
    """Run a sequential smoke/first block and return measured facts."""

    _validate_minimal_config(config)
    if type(start_seed) is not int or start_seed < 0:
        raise ValueError("start_seed must be a nonnegative integer")
    if type(deals) is not int or deals <= 0:
        raise ValueError("deals must be a positive integer")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and (nsfp is not None or solver is not None):
        raise ValueError("injected observer/solver are supported only with one worker")

    torch.set_num_threads(1)
    started = time.perf_counter()
    seeds = [start_seed + index for index in range(deals)]
    if workers == 1:
        observer = nsfp or FrozenNSFP.load(
            Path(config.nsfp.path), config.nsfp.sha256, torch.device("cpu")
        )
        terminal_solver = solver or ExactDoubleDummyCppFastestSolver()
        if hasattr(terminal_solver, "native_available") and not terminal_solver.native_available:
            raise RuntimeError("native terminal solver is unavailable")
        generated = [
            generate_hybrid_deal(seed, observer, terminal_solver) for seed in seeds
        ]
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            workers,
            initializer=_initialize_worker,
            initargs=(config.nsfp.path, config.nsfp.sha256),
        ) as pool:
            generated = pool.map(_generate_worker, seeds, chunksize=1)
    arrays = save_hybrid_npz(output, generated)
    wall_seconds = time.perf_counter() - started
    solver_calls = sum(deal.solver_calls for deal in generated)
    legal_targets = arrays.targets[arrays.masks.astype(bool)]
    return {
        "ok": True,
        "schema": "minimal-hybrid-generation-v1",
        "start_seed": start_seed,
        "deals": deals,
        "workers": workers,
        "rows": int(arrays.features.shape[0]),
        "solver_calls": solver_calls,
        "solver_calls_per_deal": solver_calls / deals,
        "wall_seconds": wall_seconds,
        "deals_per_hour": deals / wall_seconds * 3600.0,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "target_min": float(np.min(legal_targets)),
        "target_median": float(np.median(legal_targets)),
        "target_max": float(np.max(legal_targets)),
        "center_histogram": {
            str(index): int(np.count_nonzero(arrays.centers == index))
            for index in range(14)
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_bidder/base.yaml"),
    )
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--deals", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = BidderConfig.load(arguments.config)
        report = generate_to_npz(
            config,
            start_seed=arguments.start_seed,
            deals=arguments.deals,
            output=arguments.output,
            workers=arguments.workers,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
