"""Nil player card selection logic for the Spades AI.

A nil bidder wants to take zero tricks, so every decision is inverted:
- Lead the smallest, most-likely-to-be-covered card.
- Follow with the smallest card in suit; when void, dump high-value cards.
"""
from __future__ import annotations

from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.state import GameState
from spades_ai.players.rule_based_v2.helpers import cards_remaining_in_suit


def nil_player_lead(legal: frozenset[Card], state: GameState) -> Card:
    """Choose a lead card for a nil bidder.

    Strategy: score each card so that low rank and a suit with many cards
    remaining (opponents will cover) is preferred.  Spades are penalised
    heavily because leading spades risks being taken by our own low spade.
    """
    def _score(card: Card) -> float:
        # Lower score → more desirable for nil
        rank_penalty = card.rank.value  # lower rank = lower penalty (good)
        suit_penalty = 50.0 if card.suit == Suit.SPADES else 0.0
        # Reward suits where opponents still have lots of cards (they'll cover)
        remaining = cards_remaining_in_suit(card.suit, state)
        coverage_bonus = -remaining  # more remaining → lower score (good)
        return rank_penalty + suit_penalty + coverage_bonus

    return min(legal, key=_score)


def nil_player_follow(
    legal: frozenset[Card],
    led_suit: Suit,
    state: GameState,
) -> Card:
    """Choose a follow card for a nil bidder.

    Strategy:
    - Has led suit → play the smallest card (stay under the winner).
    - Void in led suit, has non-spades → dump the highest non-spade
      (get rid of dangerous winners before they can take tricks).
    - Only spades left → play the smallest spade to minimise risk.
    """
    in_suit = [c for c in legal if c.suit == led_suit]
    if in_suit:
        return min(in_suit, key=lambda c: c.rank)

    non_spades = [c for c in legal if c.suit != Suit.SPADES]
    if non_spades:
        # Dump highest non-spade: get rid of potential winners
        return max(non_spades, key=lambda c: c.rank)

    # Only spades available — play the smallest to minimise trump waste
    spades = [c for c in legal if c.suit == Suit.SPADES]
    return min(spades, key=lambda c: c.rank)
