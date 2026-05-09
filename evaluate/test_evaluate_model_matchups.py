"""Smoke test for the cross-repo Spades evaluation bridge.

File purpose:
- Verify that the GO-MCTS bridge can convert states, normalize bids, and run
  a short full hand with mixed local and collaborator players.

Function input/output summary:
- main() -> None
    Input: none.
    Output: raises on failure; prints a compact success message on pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

from adapters import GoPlayerAdapter, OurHandStrengthMCTSPlayer
from bridge import normalize_bid_for_legal_options, to_go_state
from models import RandomPlayer, RuleBasedPlayer
from strategy.spades_match_runner import SpadesMatchRunner
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig
from trick_taking.games.spades import SpadesRules


def main() -> None:
    """Run a small end-to-end smoke test for the evaluation bridge.

    Input:
    - none.

    Output:
    - Assertions on state conversion and a one-hand run; prints success on pass.
    """
    rules = SpadesRules(enable_nil=True, enable_blind_nil=True)
    # Keep the full-hand smoke run fast: use collaborator random/rule players.
    players = [
        GoPlayerAdapter(RandomPlayer(seed=5)),
        GoPlayerAdapter(RuleBasedPlayer()),
        GoPlayerAdapter(RandomPlayer(seed=7)),
        GoPlayerAdapter(RuleBasedPlayer()),
    ]

    runner = SpadesMatchRunner(players=players, seed=11, verbose=False, rules=rules)
    result = runner.play_game()
    assert sum(result.tricks_won) == 13, "full hand did not finish correctly"

    converted_state = to_go_state(runner.state)
    assert converted_state.current_player == runner.state.turn, "current player conversion mismatch"
    assert len(converted_state.completed_tricks) == len(runner.state.trick_history), "completed tricks mismatch"

    # Also validate our local hand-strength bidding adapter input contract.
    bid_tester = OurHandStrengthMCTSPlayer(
        config=TruncatedMCTSConfig(
            exact_threshold=0,
            leaf_threshold=0,
            simulations_per_action=1,
            checkpoint_path=None,
        )
    )
    bid_tester.start_game(position=0, hand=list(runner.state.hands[0]), num_players=4)
    bid_pick = bid_tester.place_bid(["nil", "bid_1", "bid_2", "bid_3"], {"state": runner.state})
    assert bid_pick in {"nil", "bid_1", "bid_2", "bid_3"}, "hand_strength bid adapter output invalid"

    legal_bids = ["nil", "bid_1", "bid_2", "bid_3"]
    assert normalize_bid_for_legal_options(0, legal_bids) == "nil"
    assert normalize_bid_for_legal_options(2, legal_bids) == "bid_2"

    print("evaluate/test_evaluate_model_matchups.py passed")


if __name__ == "__main__":
    main()
