"""
Hearts game rules implementation.

Paper reference: Table 2 — Hearts parameters (12, [0,91], 2, 500)
- 4 players, standard 52-card deck, 13 cards per hand
- No trump suit
- No bidding, no teams (individual play)
- Card points: each Heart = 1, Queen of Spades = 13
- Hearts must be broken before leading
- Shoot the moon: if one player takes all 26 points, they get 0
  and everyone else gets 26
- Objective: minimize points (lowest score wins)
"""

from __future__ import annotations

from typing import Any

from trick_taking.card import Card, Rank, RankOrder, Suit, STANDARD_RANK_ORDER
from trick_taking.deck import DeckConfig, STANDARD_52
from trick_taking.game_rules import GameRules
from trick_taking.game_state import GameState


# Queen of Spades for quick comparison
_QUEEN_OF_SPADES = Card(Suit.SPADES, Rank.QUEEN)


class HeartsRules(GameRules):
    """
    Full Hearts rules implementing the paper's Fig. 2 interface.

    Hearts is one of the 8 games covered by the paper.
    Key characteristics:
    - No trump, no bidding
    - Individual play (no teams)
    - Point cards: Hearts (+1 each), Queen of Spades (+13)
    - Hearts must be broken before leading
    - Shoot the moon: take all 26 → 0 for you, +26 for everyone else
    - Goal: minimize points
    """

    # ─── Identity ────────────────────────────────────────────────────

    @property
    def game_name(self) -> str:
        return "Hearts"

    @property
    def num_players(self) -> int:
        return 4

    @property
    def deck_config(self) -> DeckConfig:
        return STANDARD_52

    @property
    def cards_per_hand(self) -> int:
        return 13

    # ─── Card ordering ───────────────────────────────────────────────

    def rank_order(self) -> RankOrder:
        return STANDARD_RANK_ORDER

    # ─── Trump ───────────────────────────────────────────────────────

    def trump_mask(self, state: GameState) -> set[Suit] | None:
        """Hearts has no trump suit."""
        return None

    # ─── Teams ───────────────────────────────────────────────────────

    def set_team(self, state: GameState) -> list[int]:
        """Individual play — each player is their own team."""
        return [0, 1, 2, 3]

    # ─── Scoring ─────────────────────────────────────────────────────

    def points_card(self, card: Card) -> int:
        """
        Paper: "points_card — the score value of a card in a game"
        Hearts: each Heart = 1 point, Queen of Spades = 13 points.
        """
        if card.suit == Suit.HEARTS:
            return 1
        if card == _QUEEN_OF_SPADES:
            return 13
        return 0

    def score(self, state: GameState) -> list[float]:
        """
        Hearts scoring with shoot-the-moon rule.
        Paper: "to minimize the number of hearts in the tricks"

        Points are BAD in Hearts. We return negative points so that
        the GameResult.winner (max score) is the player with fewest points.
        """
        raw_points = [0.0] * 4
        for pid in range(4):
            for card in state.cards_won[pid]:
                raw_points[pid] += self.points_card(card)

        # Shoot the moon: one player took all 26 points
        for pid in range(4):
            if raw_points[pid] == 26:
                for other in range(4):
                    if other == pid:
                        raw_points[other] = 0
                    else:
                        raw_points[other] = 26
                break

        # Return negative (fewer penalty points = higher score)
        return [-p for p in raw_points]

    # ─── Legal plays ─────────────────────────────────────────────────

    def playable(self, state: GameState, hand: list[Card],
                 player_id: int) -> list[Card]:
        """
        Hearts follow-suit rules:
        1. Leading: can't lead Hearts unless broken (or only have Hearts)
        2. Following: must follow lead suit if possible
        3. First trick: can't play Hearts or Queen of Spades (optional rule)
        """
        if not hand:
            return []

        table = state.table_cards
        hearts_broken = state.trump_broken  # We reuse trump_broken for "hearts broken"

        if not table:
            # Leading
            if not hearts_broken:
                non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
                if non_hearts:
                    return non_hearts
            return list(hand)
        else:
            # Following
            lead_suit = table[0][1].suit
            suit_cards = [c for c in hand if c.suit == lead_suit]
            if suit_cards:
                return suit_cards
            # Void in lead suit — can play anything
            return list(hand)

    # ─── End condition ───────────────────────────────────────────────

    def end_trickgame(self, state: GameState) -> bool:
        return state.tricks_played >= 13

    # ─── Trick winner ────────────────────────────────────────────────

    def winner_trick(self, state: GameState) -> int:
        """
        No trump in Hearts. Highest card of lead suit wins.
        We override to ensure no trump logic is applied.
        """
        table = state.table_cards
        if not table:
            raise ValueError("No cards on table")

        rank_ord = self.rank_order()
        lead_suit = table[0][1].suit

        best_pid = table[0][0]
        best_rank_strength = rank_ord.strength(table[0][1].rank)

        for pid, card in table[1:]:
            if card.suit == lead_suit:
                strength = rank_ord.strength(card.rank)
                if strength > best_rank_strength:
                    best_pid = pid
                    best_rank_strength = strength

        # Check if hearts were broken by this trick
        for _, card in table:
            if card.suit == Suit.HEARTS:
                state.trump_broken = True
                break

        return best_pid
