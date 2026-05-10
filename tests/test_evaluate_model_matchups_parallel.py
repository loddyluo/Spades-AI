"""Parallel evaluation regression test for matchup runner.

File purpose:
- Verify that `evaluate.evaluate_model_matchups.run_evaluation` returns the same
  results when run serially and when run through the new process-pool path.
- Keep the workload intentionally small so the regression stays fast.

Function input/output summary:
- build_args() -> argparse.Namespace
    Input: none.
    Output: a fully populated argument namespace used by the evaluation runner.
- main() -> None
    Input: none.
    Output: runs serial and parallel evaluations and asserts their outputs match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import run_evaluation


def build_args() -> argparse.Namespace:
    """Build a small deterministic matchup configuration.

    Input:
    - None.

    Output:
    - An argparse namespace with the fields expected by `run_evaluation`.
    """
    return argparse.Namespace(
        seed=3786,
        num_games=2,
        output="",
        disable_nil=False,
        disable_blind_nil=True,
        p0="our_mcts",
        p1="go_rule",
        p2="our_mcts",
        p3="go_rule",
        device="cpu",
        our_checkpoint="result/mlp_test_3.pth",
        our_exact_threshold=30,
        our_leaf_threshold=24,
        our_simulations_per_action=1,
        our_exploration_constant=1.5,
        our_policy_temperature=1.0,
        our_value_scale=25.0,
        num_workers=1,
        torch_num_threads=1,
        torch_num_interop_threads=1,
        mp_start_method="fork",
        go_pv_checkpoint="",
        go_bid_checkpoint="",
        go_mcts_runs=100,
        go_mcts_steps=5,
        go_mcts_c=0.3,
        go_mcts_mu=0.01,
        go_mcts_threshold=0.05,
        go_argmax_threshold=0.05,
        trace_log_dir="",
    )


def main() -> None:
    """Run the serial-vs-parallel regression check.

    Input:
    - None.

    Output:
    - Asserts that both execution modes return identical evaluation results.
    """
    serial_args = build_args()
    serial_result = run_evaluation(serial_args)

    parallel_args = build_args()
    parallel_args.num_workers = 2
    parallel_result = run_evaluation(parallel_args)

    assert serial_result == parallel_result, "parallel evaluation changed the matchup result"
    print("parallel evaluation matches serial evaluation")


if __name__ == "__main__":
    main()
