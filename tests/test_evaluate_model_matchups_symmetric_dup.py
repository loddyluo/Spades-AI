"""Symmetry-duplicate regression test for matchup evaluation.

File purpose:
- Verify that `evaluate/evaluate_model_matchups.py` can expand each seed into
  two symmetric games when the switch is enabled.
- Verify that the second game swaps the odd/even seat groups so the matchup is
  evaluated under both seat assignments.
- Keep the test lightweight by using rule-based players for all seats.

Function input/output summary:
- build_args(symmetric_seat_swap: int) -> argparse.Namespace
    Input: 0 to disable the duplicate run, 1 to enable it.
    Output: a minimal CLI namespace for `run_evaluation`.
- test_symmetric_seat_swap_expands_each_seed_twice() -> None
    Input: none.
    Output: asserts that the duplicate run produces two games per seed with
    swapped seat layouts.
- main() -> None
    Input: none.
    Output: runs the regression assertion and prints a success message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import run_evaluation


def build_args(symmetric_seat_swap: int) -> argparse.Namespace:
    """Build a minimal evaluation configuration.

    Input:
    - symmetric_seat_swap: 0 disables the duplicate run, 1 enables it.

    Output:
    - An argparse namespace compatible with `run_evaluation`.
    """
    return argparse.Namespace(
        seed=123,
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
        our_number_of_exact_solvers=5,
        our_exploration_constant=1.5,
        our_policy_temperature=1.0,
        our_value_scale=25.0,
        symmetric_seat_swap=symmetric_seat_swap,
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


def test_symmetric_seat_swap_expands_each_seed_twice() -> None:
    """Check that one seed becomes two symmetric games when enabled.

    Input:
    - none.

    Output:
    - Raises AssertionError if the duplicate run is not expanded correctly or
      if the swapped seat layout is not reflected in the game payloads.
    """
    original_layout = ["go_rule", "go_random", "go_rule", "go_random"]
    swapped_layout = ["go_random", "go_rule", "go_random", "go_rule"]

    disabled = run_evaluation(build_args(0))
    assert disabled["num_games"] == 1
    assert disabled["num_game_pairs"] == 1
    assert len(disabled["games"]) == 1
    assert disabled["games"][0]["seat_specs"] == original_layout

    enabled = run_evaluation(build_args(1))
    assert enabled["num_games"] == 2
    assert enabled["num_game_pairs"] == 1
    assert len(enabled["games"]) == 2
    assert enabled["games"][0]["seat_specs"] == original_layout
    assert enabled["games"][1]["seat_specs"] == swapped_layout


def main() -> None:
    """Run the symmetry-duplicate regression check.

    Input:
    - none.

    Output:
    - Prints a short success message when the assertions pass.
    """
    test_symmetric_seat_swap_expands_each_seed_twice()
    print("symmetric duplicate test passed")


if __name__ == "__main__":
    main()
