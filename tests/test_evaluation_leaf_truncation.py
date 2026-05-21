"""Regression tests for the MCTS leaf truncation wiring.

These tests check the two evaluation modes requested by the user:
- exact leaf truncation at a custom remaining-card threshold
- MLP leaf truncation with the bid-specific full-info models
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_our_mcts_vs_rule_v2 import build_runtime
from strategy.truncated_mcts_strategy import SearchNode, TruncatedMCTSConfig, TruncatedMCTSStrategy


def _make_args(**overrides):
    defaults = dict(
        seed=0,
        num_games=1,
        output="",
        disable_nil=False,
        disable_blind_nil=False,
        p0="our_mcts",
        p1="go_rule_2",
        p2="our_mcts",
        p3="go_rule_2",
        device="cpu",
        our_exact_threshold=24,
        our_leaf_threshold=24,
        our_use_exact_leaf_solver=False,
        our_exact_leaf_threshold=24,
        our_simulations_per_action=50,
        our_number_of_exact_solvers=100,
        symmetric_seat_swap=0,
        num_workers=0,
        torch_num_threads=1,
        torch_num_interop_threads=1,
        mp_start_method="fork",
        our_exploration_constant=25.0,
        our_policy_temperature=1.0,
        our_mcts_determinization_count=8,
        our_value_scale=25.0,
        our_full_info_bid0_checkpoint="result/fullinfo_bid0_9.pth",
        our_full_info_bidpos_checkpoint="result/fullinfo_bidpos_9.pth",
        go_pv_checkpoint="",
        go_bid_checkpoint="",
        bid_checkpoint="",
        go_mcts_runs=100,
        go_mcts_steps=5,
        go_mcts_c=0.3,
        go_mcts_mu=0.01,
        go_mcts_threshold=0.05,
        go_argmax_threshold=0.05,
        trace_log_dir="",
        profile_breakdown=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class LeafTruncationWiringTest(unittest.TestCase):
    def test_runtime_wires_exact_leaf_solver_flags(self) -> None:
        args = _make_args(our_use_exact_leaf_solver=True, our_exact_leaf_threshold=20)
        runtime = build_runtime(args)

        self.assertTrue(runtime.local_mcts_config.exact_leaf_solver)
        self.assertEqual(runtime.local_mcts_config.exact_leaf_threshold, 20)
        self.assertEqual(runtime.local_mcts_config.leaf_threshold, 24)

    def test_exact_leaf_branch_is_used_when_enabled(self) -> None:
        strategy = TruncatedMCTSStrategy(
            TruncatedMCTSConfig(
                exact_threshold=24,
                leaf_threshold=24,
                exact_leaf_solver=True,
                exact_leaf_threshold=20,
                use_determinization=False,
                checkpoint_path=None,
                full_info_bid0_checkpoint=None,
                full_info_bidpos_checkpoint=None,
            )
        )

        calls: list[str] = []
        strategy._is_terminal = lambda state: False  # type: ignore[method-assign]
        strategy._remaining_cards = lambda state: 20  # type: ignore[method-assign]
        strategy._exact_leaf_value = lambda state: calls.append("exact") or 123.0  # type: ignore[method-assign]
        strategy._leaf_value = lambda state: calls.append("mlp") or 456.0  # type: ignore[method-assign]

        node = SearchNode(state=object())
        value = strategy._run_simulation(node, skip_determinization=True)

        self.assertEqual(value, 123.0)
        self.assertEqual(calls, ["exact"])
        self.assertEqual(node.visits, 1)
        self.assertEqual(node.value_sum, 123.0)

    def test_mlp_leaf_branch_is_used_when_exact_solver_disabled(self) -> None:
        strategy = TruncatedMCTSStrategy(
            TruncatedMCTSConfig(
                exact_threshold=24,
                leaf_threshold=24,
                exact_leaf_solver=False,
                exact_leaf_threshold=20,
                use_determinization=False,
                checkpoint_path=None,
                full_info_bid0_checkpoint=None,
                full_info_bidpos_checkpoint=None,
            )
        )

        calls: list[str] = []
        strategy._is_terminal = lambda state: False  # type: ignore[method-assign]
        strategy._remaining_cards = lambda state: 24  # type: ignore[method-assign]
        strategy._exact_leaf_value = lambda state: calls.append("exact") or 123.0  # type: ignore[method-assign]
        strategy._leaf_value = lambda state: calls.append("mlp") or 456.0  # type: ignore[method-assign]

        node = SearchNode(state=object())
        value = strategy._run_simulation(node, skip_determinization=True)

        self.assertEqual(value, 456.0)
        self.assertEqual(calls, ["mlp"])
        self.assertEqual(node.visits, 1)
        self.assertEqual(node.value_sum, 456.0)

    def test_full_info_model_selection_follows_bid_value(self) -> None:
        strategy = TruncatedMCTSStrategy(TruncatedMCTSConfig(checkpoint_path=None))
        bid0_model = object()
        bidpos_model = object()
        strategy.full_info_bid0_model = bid0_model
        strategy.full_info_bidpos_model = bidpos_model

        nil_state = SimpleNamespace(turn=0, max_bid=["nil", "bid_3", "bid_4", "bid_5"])
        pos_state = SimpleNamespace(turn=1, max_bid=["nil", "bid_3", "bid_4", "bid_5"])

        self.assertIs(strategy._select_full_info_model(nil_state), bid0_model)
        self.assertIs(strategy._select_full_info_model(pos_state), bidpos_model)


if __name__ == "__main__":
    unittest.main()