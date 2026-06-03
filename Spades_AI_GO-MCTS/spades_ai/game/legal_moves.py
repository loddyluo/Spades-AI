from __future__ import annotations
from spades_ai.game.card import Card, Suit


def get_legal_moves(
    hand: frozenset[Card],
    led_suit: Suit | None,
    spades_broken: bool,
    is_leading: bool,
) -> frozenset[Card]:
    if is_leading:
        return _legal_leads(hand, spades_broken)
    return _legal_follows(hand, led_suit)


def _legal_leads(hand: frozenset[Card], spades_broken: bool) -> frozenset[Card]:
    if spades_broken:
        return hand
    non_spades = frozenset(c for c in hand if c.suit != Suit.SPADES)
    return non_spades if non_spades else hand


def _legal_follows(hand: frozenset[Card], led_suit: Suit | None) -> frozenset[Card]:
    if led_suit is None:
        return hand
    matching = frozenset(c for c in hand if c.suit == led_suit)
    return matching if matching else hand
