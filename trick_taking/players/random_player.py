"""
Random player — chooses uniformly at random from legal actions.

Used for testing, baseline evaluation, and self-play experiments.
Paper reference: Section 6 "Experiments" — selfplay evaluation.
"""

from __future__ import annotations

import random
from typing import Any

from trick_taking.card import Card
from trick_taking.player import AIPlayer


class RandomPlayer(AIPlayer):
    """
    Player that selects uniformly at random from legal actions.
    Useful as a baseline and for framework testing.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._position: int = -1
        self._hand: list[Card] = []

    def start_game(self, position: int, hand: list[Card],
                   num_players: int) -> None:
        self._position = position
        self._hand = list(hand)

    def place_bid(self, legal_bids: list[Any],
                  state_view: dict) -> Any:
        return self._rng.choice(legal_bids)

    def play_card(self, legal_cards: list[Card],
                  state_view: dict) -> Card:
        return self._rng.choice(legal_cards)
