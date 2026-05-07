"""
Human player — interactive console interface.

Paper reference: Section 6.2 "Competitive Play"
"We played all players interactively. For casual games, the playing
strength is acceptable and the response time small enough for swift play."
"""

from __future__ import annotations

from typing import Any

from trick_taking.card import Card
from trick_taking.player import AIPlayer


class HumanPlayer(AIPlayer):
    """
    Interactive human player using console input.
    Displays game state and prompts for card/bid selection.
    """

    def __init__(self, name: str = "Human") -> None:
        self._name = name
        self._position: int = -1
        self._hand: list[Card] = []

    def start_game(self, position: int, hand: list[Card],
                   num_players: int) -> None:
        self._position = position
        self._hand = list(hand)
        print(f"\n{'='*50}")
        print(f"  {self._name} (Player {position})")
        print(f"  Hand: {_format_hand(hand)}")
        print(f"{'='*50}")

    def place_bid(self, legal_bids: list[Any],
                  state_view: dict) -> Any:
        print(f"\n[Bidding] Your hand: {_format_hand(state_view['hand'])}")
        print(f"  Bids so far: {state_view['bids']}")
        print(f"  Legal bids: {legal_bids}")
        while True:
            choice = input("  Your bid: ").strip()
            if choice in [str(b) for b in legal_bids]:
                # Find matching bid
                for b in legal_bids:
                    if str(b) == choice:
                        return b
            print(f"  Invalid! Choose from: {legal_bids}")

    def bid_placed(self, player_id: int, bid_value: Any) -> None:
        prefix = "  >> You" if player_id == self._position else f"  >> Player {player_id}"
        print(f"{prefix} bid: {bid_value}")

    def set_teams(self, teams: list[int], bids: list[Any]) -> None:
        print(f"\n[Teams] {teams}")

    def play_card(self, legal_cards: list[Card],
                  state_view: dict) -> Card:
        print(f"\n[Trick] Table: {_format_trick(state_view['table_cards'])}")
        print(f"  Trump broken: {state_view['trump_broken']}")
        print(f"  Your hand: {_format_hand(state_view['hand'])}")
        print(f"  Legal plays ({len(legal_cards)}): {_format_hand(legal_cards)}")

        while True:
            choice = input("  Play card (e.g. A♠ or SA): ").strip()
            for card in legal_cards:
                if choice == str(card) or choice == card.repr_short():
                    return card
            # Try parsing as index
            try:
                idx = int(choice)
                if 0 <= idx < len(legal_cards):
                    return legal_cards[idx]
            except ValueError:
                pass
            print(f"  Invalid! Enter card name or index 0-{len(legal_cards)-1}")

    def card_played(self, player_id: int, card: Card) -> None:
        prefix = "  You" if player_id == self._position else f"  P{player_id}"
        print(f"  {prefix} played {card}")


def _format_hand(cards: list) -> str:
    """Format a hand of cards for display."""
    return " ".join(str(c) for c in cards)


def _format_trick(table: list) -> str:
    """Format current trick cards."""
    if not table:
        return "(empty)"
    return " ".join(f"P{pid}:{card}" for pid, card in table)
