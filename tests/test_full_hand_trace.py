"""Run a single full deal and print bidding, plays, and scoring details.

This script is intended as a human-run debug helper rather than an automated
assertion-style unit test. Run with `python tests/test_full_hand_trace.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running the script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.spades_player_programs import RandomSpadesPlayer
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules


def run_trace(seed: int = 0) -> None:
    players = [RandomSpadesPlayer(seed + i) for i in range(4)]
    rules = SpadesRules(enable_nil=True, enable_blind_nil=True)

    runner = SpadesMatchRunner(players=players, seed=seed, verbose=True, rules=rules)
    result = runner.play_game()

    state = runner.state

    print("\n=== Summary Trace ===")
    print(f"Seed: {seed}")
    print("Initial hands:")
    for pid, hand in enumerate(state.hands):
        print(f"  Player {pid} hand ({len(hand)}): {hand}")

    print("\nBids (stack order):")
    for b in state.bids:
        print(f"  player={b.player_id} value={b.value} pass={b.is_pass}")

    print("\nResolved max bids:")
    for pid, mb in enumerate(state.max_bid):
        print(f"  Player {pid}: {mb}")

    print("\nTrick history (in order):")
    for ti, tr in enumerate(state.trick_history):
        cards_desc = ", ".join(f"P{pid}:{card}" for pid, card in tr.cards)
        print(f"  Trick {ti}: leader={tr.leader} winner={tr.winner} cards=[{cards_desc}]")

    print("\nPer-player tricks won:")
    for pid, t in enumerate(state.tricks_won):
        print(f"  Player {pid}: {t}")

    print("\nScores from runner.result:")
    print(f"  raw scores: {result.scores}")

    print("\nScores computed by rules.score(state):")
    print(f"  {rules.score(state)}")


if __name__ == "__main__":
    run_trace(seed=0)
