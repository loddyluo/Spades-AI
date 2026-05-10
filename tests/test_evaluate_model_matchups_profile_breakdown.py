"""Regression tests for matchup profile-breakdown output.

File purpose:
- Verify that `evaluate/evaluate_model_matchups.py` exposes the
  `--profile-breakdown` plumbing via runtime args and includes per-game timing
  details only when enabled.

Function input/output summary:
- build_args(profile_breakdown: int) -> argparse.Namespace
    Input: 0 disables per-game breakdown, 1 enables it.
    Output: a minimal CLI namespace for `run_evaluation`.
- test_profile_breakdown_toggle() -> None
    Input: none.
    Output: asserts that breakdown payload exists only when enabled and has
    expected structure.
- main() -> None
    Input: none.
    Output: runs assertions and prints a short success message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import run_evaluation


def build_args(profile_breakdown: int) -> argparse.Namespace:
    """Build a minimal evaluation configuration.

    Input:
    - profile_breakdown: 0 disables breakdown, 1 enables it.

    Output:
    - An argparse namespace compatible with `run_evaluation`.
    """
    return argparse.Namespace(
        seed=321,
        num_games=1,
        output="",
        disable_nil=False,
        disable_blind_nil=True,
        p0="go_rule",
        p1="go_random",
        p2="go_rule",
        p3="go_random",
        device="cpu",
        our_checkpoint="",
        our_exact_threshold=24,
        our_leaf_threshold=24,
        our_simulations_per_action=1,
        our_number_of_exact_solvers=3,
        our_exploration_constant=1.5,
        our_policy_temperature=1.0,
        our_value_scale=25.0,
        symmetric_seat_swap=0,
        num_workers=1,
        torch_num_threads=1,
        torch_num_interop_threads=1,
        mp_start_method="fork",
        go_pv_checkpoint="",
        go_bid_checkpoint="",
        go_mcts_runs=50,
        go_mcts_steps=3,
        go_mcts_c=0.3,
        go_mcts_mu=0.01,
        go_mcts_threshold=0.05,
        go_argmax_threshold=0.05,
        trace_log_dir="",
        profile_breakdown=profile_breakdown,
    )


def test_profile_breakdown_toggle() -> None:
    """Check that profile breakdown can be toggled from args.

    Input:
    - none.

    Output:
    - Raises AssertionError if the payload presence/shape is incorrect.
    """
    disabled = run_evaluation(build_args(0))
    assert "profile_breakdown" not in disabled["games"][0]

    enabled = run_evaluation(build_args(1))
    breakdown = enabled["games"][0].get("profile_breakdown")
    assert breakdown is not None
    assert float(breakdown["game_wall_sec"]) >= 0.0
    assert len(breakdown["seats"]) == 4
    assert len(breakdown["strategy_diagnostics"]) == 4

    first_seat = breakdown["seats"][0]
    for key in ["bid_calls", "play_calls", "bid_elapsed_sec", "play_elapsed_sec", "bid_max_sec", "play_max_sec"]:
        assert key in first_seat


def main() -> None:
    """Run profile-breakdown regression checks.

    Input:
    - none.

    Output:
    - Prints a short success message when assertions pass.
    """
    test_profile_breakdown_toggle()
    print("profile breakdown tests passed")


if __name__ == "__main__":
    main()
