"""Regression checks for exact-branch C++ solver usage and consistency.

File purpose:
- Verify that `TruncatedMCTSStrategy` prefers the C++ opt1 exact solver when
  native support is available.
- Verify that C++ opt1 `solve_with_q` remains numerically consistent with the
  Python reference exact solver on deterministic test states.

Function input/output summary:
- _build_states(target_remaining: int, seeds: list[int]) -> list[GameState]
    Input: fixed remaining-card count and deterministic seeds.
    Output: list of reproducible test states.
- _q_map(result: dict) -> dict[int, float]
    Input: `solve_with_q` result dictionary.
    Output: card_id -> q_value map for stable comparison.
- test_cpp_opt1_consistency_against_python_reference() -> None
    Input: none.
    Output: asserts value/q consistency for multiple deterministic states.
- test_strategy_prefers_cpp_solver_when_available() -> None
    Input: none.
    Output: asserts strategy exact solver selection behavior.
- main() -> None
    Input: none.
    Output: runs checks and prints a short status line.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import build_state_with_remaining_cards
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy
from trick_taking.game_state import GameState
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver


def _build_states(target_remaining: int, seeds: list[int]) -> list[GameState]:
    """Build deterministic test states.

    Input:
    - target_remaining: remaining-card count for each state.
    - seeds: deterministic seeds used for reproducible state generation.

    Output:
    - List of GameState objects.
    """
    return [build_state_with_remaining_cards(target_remaining=target_remaining, seed=seed) for seed in seeds]


def _q_map(result: dict) -> dict[int, float]:
    """Convert `solve_with_q` action dictionary to card_id keyed mapping.

    Input:
    - result: dictionary returned by `solve_with_q`.

    Output:
    - `card_id -> q_value` dictionary.
    """
    out: dict[int, float] = {}
    for action, value in result.get("action_q_values", {}).items():
        out[action.card_id] = float(value)
    return out


def test_cpp_opt1_consistency_against_python_reference() -> None:
    """Check C++ opt1 exact solve consistency against Python reference.

    Input:
    - none.

    Output:
    - Raises AssertionError on any value/Q mismatch beyond tolerance.
    """
    cpp_solver = ExactDoubleDummyCppOpt1Solver()
    if not cpp_solver.native_available:
        print("skip: cpp opt1 solver native library unavailable")
        return

    py_solver = ExactDoubleDummySolver()
    states = _build_states(target_remaining=8, seeds=[3, 7, 11])

    for idx, state in enumerate(states):
        cpp_res = cpp_solver.solve_with_q(state)
        py_res = py_solver.solve_with_q(state)

        assert math.isclose(float(cpp_res["value"]), float(py_res["value"]), rel_tol=1e-8, abs_tol=1e-8), (
            f"value mismatch at state#{idx}: cpp={cpp_res['value']} py={py_res['value']}"
        )

        cpp_q = _q_map(cpp_res)
        py_q = _q_map(py_res)
        assert set(cpp_q.keys()) == set(py_q.keys()), f"action set mismatch at state#{idx}"
        for aid in sorted(cpp_q.keys()):
            assert math.isclose(cpp_q[aid], py_q[aid], rel_tol=1e-8, abs_tol=1e-8), (
                f"q mismatch at state#{idx} action_id={aid}: cpp={cpp_q[aid]} py={py_q[aid]}"
            )


def test_strategy_prefers_cpp_solver_when_available() -> None:
    """Check strategy exact solver selection policy.

    Input:
    - none.

    Output:
    - Raises AssertionError if solver selection does not follow availability.
    """
    strategy = TruncatedMCTSStrategy(
        TruncatedMCTSConfig(
            exact_threshold=24,
            leaf_threshold=24,
            simulations_per_action=1,
            determinization_count=1,
            use_determinization=False,
            checkpoint_path=None,
        )
    )

    cpp_probe = ExactDoubleDummyCppOpt1Solver()
    if cpp_probe.native_available:
        assert isinstance(strategy.exact_solver, ExactDoubleDummyCppOpt1Solver)
    else:
        assert isinstance(strategy.exact_solver, ExactDoubleDummySolver)


def main() -> None:
    """Run all regression checks.

    Input:
    - none.

    Output:
    - Prints success when all checks pass.
    """
    test_cpp_opt1_consistency_against_python_reference()
    test_strategy_prefers_cpp_solver_when_available()
    print("exact cpp consistency tests passed")


if __name__ == "__main__":
    main()
