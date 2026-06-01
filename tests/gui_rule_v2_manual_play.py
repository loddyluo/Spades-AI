"""Manual smoke test for a single Spades hand with three rule_v2 seats.

Input:
- A random seed, optional human seat, and optional auto-human fallback.

Output:
- A step-by-step text trace of the hand, including bids, tricks, and scores.

This script is intended to validate the same player wiring that the GUI will
use: three seats run the collaborator's `rule_based_v2` logic and one seat is
played by a human from the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLAB_ROOT = REPO_ROOT / "Spades_AI_GO-MCTS"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLAB_ROOT))

from spades_ai.game.card import Card
from spades_ai.game.deck import deal_hands
from spades_ai.game.legal_moves import get_legal_moves
from spades_ai.game.scoring import BidType, compute_hand_scores, PlayerResult
from spades_ai.game.state import Bid, GameState, Phase
from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as RuleBasedPlayerV2


def parse_args() -> argparse.Namespace:
    """Parse the smoke-test command line.

    Input:
    - `sys.argv`.

    Output:
    - Parsed arguments for seed, human seat, and fallback mode.
    """
    parser = argparse.ArgumentParser(description="Manual smoke test for rule_v2 vs one human seat.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the deal")
    parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1, 2, 3], help="Seat controlled by the user")
    parser.add_argument("--auto-human", action="store_true", help="Auto-pick the first legal action when stdin is not interactive")
    return parser.parse_args()


def format_card(card: Card) -> str:
    """Format a card for human-readable output.

    Input:
    - A `spades_ai.game.card.Card` instance.

    Output:
    - A compact string such as `A♠`.
    """
    return str(card)


def format_hand(hand: frozenset[Card]) -> str:
    """Format a hand for display.

    Input:
    - One seat's hand as a frozenset of cards.

    Output:
    - A sorted, space-separated card string.
    """
    return " ".join(sorted((format_card(card) for card in hand), key=lambda text: (text[-1], text[:-1])))


def prompt_choice(prompt: str, legal: list[str], auto_human: bool) -> str:
    """Collect a human action from stdin or use the fallback.

    Input:
    - `prompt`: text shown to the user.
    - `legal`: the list of legal actions.
    - `auto_human`: whether to auto-pick the first legal action.

    Output:
    - The chosen action string.
    """
    if auto_human or not sys.stdin.isatty():
        choice = legal[0]
        print(f"{prompt} {choice} [auto]")
        return choice

    while True:
        choice = input(f"{prompt} ({', '.join(legal)}): ").strip()
        if choice in legal:
            return choice
        print("Illegal choice, try again.")


def make_bid_from_text(text: str) -> Bid:
    """Convert terminal input into a `Bid` record.

    Input:
    - A numeric bid string or the word `nil`.

    Output:
    - A `Bid` instance for the local engine.
    """
    lowered = text.strip().lower()
    if lowered == "nil":
        return Bid(value=0, bid_type=BidType.NIL)
    return Bid(value=int(lowered), bid_type=BidType.NORMAL)


def state_summary(state: GameState) -> str:
    """Render a compact state summary.

    Input:
    - Current `GameState` snapshot.

    Output:
    - A one-line human-readable summary.
    """
    return (
        f"phase={state.phase.name} current={state.current_player} leader={state.leader} "
        f"trick={state.trick_number} tricks_won={state.tricks_won}"
    )


def main() -> None:
    """Run one hand with a single human seat.

    Input:
    - Parsed command line arguments.

    Output:
    - A step-by-step terminal trace until the hand ends.
    """
    args = parse_args()
    ai = RuleBasedPlayerV2()
    hands = deal_hands(args.seed)
    state = GameState.new_game(hands)

    print(f"Seed: {args.seed}")
    print(f"Human seat: {args.human_seat}")
    print("Initial hands:")
    for seat, hand in enumerate(state.hands):
        tag = "HUMAN" if seat == args.human_seat else "AI"
        print(f"  P{seat} ({tag}): {format_hand(hand)}")
    print()

    while state.phase == Phase.BIDDING:
        seat = state.current_player
        if seat == args.human_seat:
            legal = ["nil"] + [str(index) for index in range(1, 14)]
            choice = prompt_choice(f"P{seat} bid", legal, args.auto_human)
            bid = make_bid_from_text(choice)
        else:
            bid = ai.choose_bid(state)
            print(f"P{seat} bid -> {bid.value if bid.bid_type == BidType.NORMAL else 'nil'}")
        state = state.place_bid(bid)
        print(state_summary(state))

    print("\nPlaying phase begins\n")
    while state.phase == Phase.PLAYING:
        seat = state.current_player
        if seat == args.human_seat:
            hand = sorted(state.hands[seat], key=lambda card: card.index)
            legal = get_legal_moves(
                hand=state.hands[seat],
                led_suit=state.led_suit,
                spades_broken=state.spades_broken,
                is_leading=len(state.current_trick_cards) == 0,
            )
            legal_codes = sorted(format_card(card) for card in legal)
            choice = prompt_choice(f"P{seat} play", legal_codes, args.auto_human)
            selected = next(card for card in hand if format_card(card) == choice)
            state = state.play_card(selected)
        else:
            card = ai.choose_card(state)
            print(f"P{seat} play -> {format_card(card)}")
            state = state.play_card(card)
        print(state_summary(state))
        if state.current_trick_cards:
            trick_text = "  trick: " + " | ".join(f"P{entry.player}:{format_card(entry.card)}" for entry in state.current_trick_cards)
            print(trick_text)
        print()

    # compute final team scores from the four players' results
    team_a, team_b = compute_hand_scores(
        [
            type("PR", (), {"bid": state.bids[0].value, "bid_type": state.bids[0].bid_type, "tricks_won": state.tricks_won[0]})(),
            type("PR", (), {"bid": state.bids[1].value, "bid_type": state.bids[1].bid_type, "tricks_won": state.tricks_won[1]})(),
            type("PR", (), {"bid": state.bids[2].value, "bid_type": state.bids[2].bid_type, "tricks_won": state.tricks_won[2]})(),
            type("PR", (), {"bid": state.bids[3].value, "bid_type": state.bids[3].bid_type, "tricks_won": state.tricks_won[3]})(),
        ]
    )
    print("Final score:")
    print(f"  Team NS: {team_a}")
    print(f"  Team EW: {team_b}")


if __name__ == "__main__":
    main()