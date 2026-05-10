"""Configuration regression test for matchup determinization settings.

File purpose:
- Verify that the evaluation CLI plumbing forwards the new exact-solver
  resampling parameter and the 24-card exact cutoff into the local MCTS
  configuration.
- Keep the test lightweight by checking the runtime object directly instead of
  running a full game.

Function input/output summary:
- build_args() -> argparse.Namespace
    Input: none.
    Output: a minimal argument namespace compatible with `build_runtime`.
- test_build_runtime_forwards_determinization_settings() -> None
    Input: none.
    Output: asserts that the runtime config reflects the new CLI defaults.
- main() -> None
    Input: none.
    Output: runs the assertion and prints a success message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import build_runtime, parse_args


def build_args() -> argparse.Namespace:
    """Build a minimal CLI namespace for runtime construction.

    Input:
    - none.

    Output:
    - An argparse.Namespace containing the fields consumed by build_runtime.
    """
    return argparse.Namespace(
        device="cpu",
        p0="our_mcts",
        p1="go_rule",
        p2="go_rule",
        p3="go_random",
        our_checkpoint="",
        our_exact_threshold=24,
        our_leaf_threshold=24,
        our_simulations_per_action=11,
        our_number_of_exact_solvers=50,
        our_exploration_constant=1.5,
        our_policy_temperature=1.0,
        our_value_scale=25.0,
        go_pv_checkpoint="",
        go_bid_checkpoint="",
        go_mcts_runs=100,
        go_mcts_steps=5,
        go_mcts_c=0.3,
        go_mcts_mu=0.01,
        go_mcts_threshold=0.05,
        go_argmax_threshold=0.05,
    )


def test_build_runtime_forwards_determinization_settings() -> None:
    """Check that build_runtime forwards the new CLI parameters.

    Input:
    - none.

    Output:
    - Raises AssertionError if the runtime config does not mirror the CLI
      determinization settings.
    """
    args = build_args()
    runtime = build_runtime(args)
    assert runtime.local_mcts_config.exact_threshold == 24
    assert runtime.local_mcts_config.leaf_threshold == 24
    assert runtime.local_mcts_config.simulations_per_action == 11
    assert runtime.local_mcts_config.determinization_count == 50


def test_parse_args_defaults_match_timing_budget() -> None:
    """Check that the evaluation CLI defaults stay aligned with timing runs.

    Input:
    - none.

    Output:
    - Raises AssertionError if the parser defaults drift back to the heavy
      settings that caused the long runtime regression.
    """
    original_argv = list(sys.argv)
    try:
        sys.argv = ["evaluate_model_matchups.py"]
        args = parse_args()
    finally:
        sys.argv = original_argv

    assert args.our_simulations_per_action == 50
    assert args.our_number_of_exact_solvers == 50
    assert args.symmetric_seat_swap == 0


def main() -> None:
    """Run the configuration regression check.

    Input:
    - none.

    Output:
    - Prints a short success message when the assertions pass.
    """
    test_build_runtime_forwards_determinization_settings()
    test_parse_args_defaults_match_timing_budget()
    print("matchup determinization config test passed")


if __name__ == "__main__":
    main()
