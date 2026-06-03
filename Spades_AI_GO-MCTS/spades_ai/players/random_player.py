"""Random player — selects bids and cards uniformly at random.

Useful as a baseline opponent and for testing the game engine.
"""
from __future__ import annotations

import random

from spades_ai.game.card import Card
from spades_ai.game.legal_moves import get_legal_moves
from spades_ai.game.scoring import BidType
from spades_ai.game.state import Bid, GameState


class RandomPlayer:
    """A player that chooses actions uniformly at random.

    Parameters
    ----------
    seed:
        Optional seed for the internal PRNG.  Two ``RandomPlayer`` instances
        created with the same seed will make identical choices given the same
        sequence of states.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_bid(self, state: GameState) -> Bid:
        """Return a random NORMAL bid in [1, 6]."""
        return Bid(value=self._rng.randint(1, 6), bid_type=BidType.NORMAL)

    def choose_card(self, state: GameState) -> Card:
        """Return a randomly chosen legal card from the current player's hand."""
        player_idx = state.current_player
        hand = state.hands[player_idx]
        is_leading = len(state.current_trick_cards) == 0
        legal = get_legal_moves(
            hand,
            led_suit=state.led_suit,
            spades_broken=state.spades_broken,
            is_leading=is_leading,
        )
        # sort for deterministic ordering before sampling
        return self._rng.choice(sorted(legal))
