"""Determinization regression tests for the truncated MCTS strategy.

File purpose:
- Verify that opponent-hand determinization preserves public information and
  hand sizes.
- Verify that both the exact-solve branch and the MCTS branch can execute with
  determinization enabled without crashing or leaking obvious private-state
  assumptions.

Function input/output summary:
- build_strategy(exact_threshold: int, leaf_threshold: int, simulations_per_action: int, determinization_count: int) -> TruncatedMCTSStrategy
    Input: search/config parameters for the strategy.
    Output: a configured strategy instance with determinization enabled.
- test_determinize_state_preserves_public_information() -> None
    Input: none.
    Output: asserts that determinization keeps observer-visible state stable and
    only resamples hidden opponent hands.
- test_choose_action_with_info_exact_branch_uses_determinization_wrapper() -> None
    Input: none.
    Output: asserts that the exact branch routes through the determinization
    wrapper and returns the wrapped result.
- test_choose_action_with_info_mcts_branch_smoke() -> None
    Input: none.
    Output: asserts that the MCTS branch can run with determinization enabled
    and that the determinization helper is actually invoked.
- main() -> None
    Input: none.
    Output: runs all assertions and prints a short success message.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import MethodType
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import build_state_with_remaining_cards
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy


def build_strategy(
    exact_threshold: int,
    leaf_threshold: int,
    simulations_per_action: int,
    determinization_count: int,
) -> TruncatedMCTSStrategy:
    """Build a strategy instance for regression tests.

    Input:
    - exact_threshold: remaining-card cutoff for the exact branch.
    - leaf_threshold: remaining-card cutoff for the MCTS leaf evaluator.
    - simulations_per_action: number of simulations per root action.
    - determinization_count: number of sampled worlds for the exact wrapper.

    Output:
    - A TruncatedMCTSStrategy configured to use determinization.
    """
    config = TruncatedMCTSConfig(
        exact_threshold=exact_threshold,
        leaf_threshold=leaf_threshold,
        simulations_per_action=simulations_per_action,
        determinization_count=determinization_count,
        use_determinization=True,
        checkpoint_path=None,
    )
    return TruncatedMCTSStrategy(config)


def test_determinize_state_preserves_public_information() -> None:
    """Check that determinization only changes hidden opponent hands.

    Input:
    - none.

    Output:
    - Raises AssertionError if observer hand, played cards, or hand sizes are
      modified by determinization.
    """
    state = build_state_with_remaining_cards(target_remaining=8, seed=7)
    observer_id = state.turn
    original_observer_hand = list(state.hands[observer_id])
    original_hand_sizes = [len(hand) for hand in state.hands]
    original_played_bitset = state.played_bitset
    original_table_cards = list(state.table_cards)
    original_trick_history = list(state.trick_history)

    strategy = build_strategy(exact_threshold=0, leaf_threshold=8, simulations_per_action=1, determinization_count=2)
    determinized = state
    strategy._determinize_state(determinized, observer_id=observer_id, rng=random.Random(1234))

    assert determinized.hands[observer_id] == original_observer_hand
    assert [len(hand) for hand in determinized.hands] == original_hand_sizes
    assert determinized.played_bitset == original_played_bitset
    assert determinized.table_cards == original_table_cards
    assert determinized.trick_history == original_trick_history

    card_ids = [card.card_id for hand in determinized.hands for card in hand]
    assert len(card_ids) == len(set(card_ids)), "determinization produced duplicate cards"


def test_choose_action_with_info_exact_branch_uses_determinization_wrapper() -> None:
    """Check that the exact branch routes through the determinization wrapper.

    Input:
    - none.

    Output:
    - Raises AssertionError if the exact branch bypasses the wrapper or fails
      to return the wrapped action.
    """
    state = build_state_with_remaining_cards(target_remaining=8, seed=11)
    strategy = build_strategy(exact_threshold=8, leaf_threshold=8, simulations_per_action=1, determinization_count=3)
    wrapped_action = state.hands[state.turn][0]
    calls: list[int] = []

    def fake_solve_with_determinization(self, input_state):
        calls.append(input_state.turn)
        return {
            "value": 1.25,
            "best_action": wrapped_action,
            "action_q_values": {wrapped_action: 1.25},
        }

    with patch.object(strategy, "_solve_with_determinization", MethodType(fake_solve_with_determinization, strategy)):
        info = strategy.choose_action_with_info(state)

    assert calls == [state.turn]
    assert info["mode"] == "exact"
    assert info["best_action"] == wrapped_action
    assert info["best_value"] == 1.25
    assert info["action_scores"]


def test_choose_action_with_info_mcts_branch_smoke() -> None:
    """Check that the MCTS branch can run with determinization enabled.

    Input:
    - none.

    Output:
    - Raises AssertionError if the MCTS path crashes, fails to invoke the
      determinization helper, or returns an illegal action.
    """
    state = build_state_with_remaining_cards(target_remaining=8, seed=13)
    strategy = build_strategy(exact_threshold=0, leaf_threshold=8, simulations_per_action=1, determinization_count=2)
    original_determinize = strategy._determinize_state
    calls: list[int] = []

    def counting_determinize(input_state, observer_id, rng=None):
        calls.append(observer_id)
        return original_determinize(input_state, observer_id, rng)

    strategy._determinize_state = counting_determinize  # type: ignore[method-assign]
    info = strategy.choose_action_with_info(state)

    assert info["mode"] == "mcts"
    assert calls, "determinization helper was not invoked in MCTS branch"
    assert info["best_action"] in state.hands[state.turn]
    assert info["action_scores"]


def test_policy_priors_are_uniform_without_model_call() -> None:
    """Check that policy priors are forced to uniform and skip model calls.

    Input:
    - none.

    Output:
    - Raises AssertionError if priors are non-uniform or if policy model
      forward is invoked.
    """
    state = build_state_with_remaining_cards(target_remaining=8, seed=19)
    strategy = build_strategy(exact_threshold=0, leaf_threshold=8, simulations_per_action=1, determinization_count=2)
    legal = strategy._legal_actions(state)

    class _GuardModel:
        def predict_policy_logits(self, _feature):
            raise AssertionError("policy model forward should not be called")

    strategy.model = _GuardModel()  # type: ignore[assignment]
    priors = strategy._policy_priors(state, legal)

    assert priors
    expected = 1.0 / len(legal)
    for action in legal:
        assert abs(priors[action.card_id] - expected) < 1e-12
    assert strategy.get_diagnostics()["policy_model_calls"] == 0


def main() -> None:
    """Run the determinization regression checks.

    Input:
    - none.

    Output:
    - Prints a short success message when all assertions pass.
    """
    test_determinize_state_preserves_public_information()
    test_choose_action_with_info_exact_branch_uses_determinization_wrapper()
    test_choose_action_with_info_mcts_branch_smoke()
    test_policy_priors_are_uniform_without_model_call()
    print("determinization tests passed")


if __name__ == "__main__":
    main()
