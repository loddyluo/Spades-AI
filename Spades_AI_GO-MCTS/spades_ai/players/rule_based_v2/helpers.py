"""Helper utilities for rule-based player logic."""
from __future__ import annotations

from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.state import GameState


def get_suit_cards(hand: frozenset[Card], suit: Suit) -> list[Card]:
    """Return cards of the given suit sorted descending by rank."""
    return sorted([c for c in hand if c.suit == suit], key=lambda c: c.rank, reverse=True)


def is_master(card: Card, suit: Suit, state: GameState) -> bool:
    """Return True if no higher card in the suit remains unplayed."""
    played: set[Card] = set()
    for trick in state.completed_tricks:
        for tc in trick.cards:
            played.add(tc.card)
    for tc in state.current_trick_cards:
        played.add(tc.card)

    for rv in range(card.rank.value + 1, 15):
        candidate = Card(Rank(rv), suit)
        if candidate not in played:
            return False
    return True


def opponent_is_void(suit: Suit, player: int, state: GameState) -> bool:
    """Return True if the given player is known to be void in the suit."""
    return suit in state.void_shown[player]


def get_opponents(player: int) -> tuple[int, int]:
    """Return the seat indices of the two opponents of player."""
    return ((player + 1) % 4, (player + 3) % 4)


def cards_remaining_in_suit(suit: Suit, state: GameState) -> int:
    """Return the count of unplayed cards in the given suit."""
    played = 0
    for trick in state.completed_tricks:
        for tc in trick.cards:
            if tc.card.suit == suit:
                played += 1
    for tc in state.current_trick_cards:
        if tc.card.suit == suit:
            played += 1
    return 13 - played
