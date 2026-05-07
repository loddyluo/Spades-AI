"""
Spades game rules implementation.

Paper reference: Table 2 — Spades parameters (20, [0,192], 30, 500)
- 4 players, standard 52-card deck, 13 cards per hand
- Fixed trump: Spades (must be broken before leading)
- Fixed teams: players 0&2 vs 1&3
- Single-round bidding: each player bids once (nil, blind nil allowed)
- Scoring: contract * 10, overtricks -9 each, nil ±50, blind nil ±100
"""

from __future__ import annotations

from typing import Any

from trick_taking.card import Card, Rank, RankOrder, Suit, STANDARD_RANK_ORDER
from trick_taking.deck import DeckConfig, STANDARD_52
from trick_taking.game_rules import GameRules
from trick_taking.game_state import GameState


class SpadesRules(GameRules):
    """
    Full Spades rules implementing the paper's Fig. 2 interface.

    Spades is one of the 8 games covered by the paper (Table 2).
    Key characteristics:
    - Fixed trump (Spades), must be broken
    - Single-round bidding with nil/blind nil
    - Partnership game (0&2 vs 1&3)
    """

    def __init__(self, enable_nil: bool = True,
                 enable_blind_nil: bool = True) -> None:
        self._enable_nil = enable_nil
        self._enable_blind_nil = enable_blind_nil

    # ─── Identity ────────────────────────────────────────────────────

    @property
    def game_name(self) -> str:
        return "Spades"

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
        return {Suit.SPADES}

    # ─── Teams ───────────────────────────────────────────────────────

    def set_team(self, state: GameState) -> list[int]:
        """Fixed partnerships: 0&2 = team 0, 1&3 = team 1."""
        return [0, 1, 0, 1]

    # ─── Bidding ─────────────────────────────────────────────────────

    @property
    def has_bidding(self) -> bool:
        return True

    def legal_bids(self, state: GameState, player_id: int) -> list[Any]:
        """
        Single-round bidding. Each player bids once.
        Legal bids: nil (if enabled), bid_1 through bid_13.
        Blind nil: only before seeing cards (first bid opportunity).
        """
        bids: list[Any] = []

        # Check if player already bid
        player_bids = [b for b in state.bids if b.player_id == player_id
                       and not b.is_pass]
        if player_bids:
            return []

        # Blind nil option (before passing on it)
        blind_passed = any(
            b.player_id == player_id and b.is_pass for b in state.bids
        )
        if self._enable_blind_nil and not blind_passed and not player_bids:
            return ["blind_nil", "pass"]

        # Normal bidding
        if self._enable_nil:
            bids.append("nil")
        for i in range(1, 14):
            bids.append(f"bid_{i}")

        return bids

    def end_bidding(self, state: GameState) -> bool:
        """Bidding ends when all 4 players have placed actual bids (not passes)."""
        actual_bids = [b for b in state.bids if not b.is_pass]
        return len(actual_bids) >= self.num_players

    def next_bid_turn(self, state: GameState) -> int:
        """
        After a pass (declining blind nil), same player bids again.
        After an actual bid, move to next player.
        """
        if state.bids and state.bids[-1].is_pass:
            return state.bids[-1].player_id  # Same player must now bid
        return (state.current_bidder + 1) % self.num_players

    # ─── Legal plays (paper's "playable") ────────────────────────────

    def playable(self, state: GameState, hand: list[Card],
                 player_id: int) -> list[Card]:
        """
        Spades follow-suit rules:
        1. Leading: can't lead spades unless broken or only have spades
        2. Following: must follow lead suit if possible, else any card
        """
        if not hand:
            return []

        table = state.table_cards

        if not table:
            # Leading
            if not state.trump_broken:
                non_spades = [c for c in hand if c.suit != Suit.SPADES]
                if non_spades:
                    return non_spades
            return list(hand)
        else:
            # Following
            lead_suit = table[0][1].suit
            suit_cards = [c for c in hand if c.suit == lead_suit]
            if suit_cards:
                return suit_cards
            return list(hand)

    # ─── End condition ───────────────────────────────────────────────

    def end_trickgame(self, state: GameState) -> bool:
        return state.tricks_played >= 13

    # ─── Scoring ─────────────────────────────────────────────────────

    def score(self, state: GameState) -> list[float]:
        """
        Spades scoring with harsh overtrick penalty:
        - Met bid: bid * 10 - overtricks * 9  (each overtrick costs 9 net)
        - Failed bid: -bid * 10
        - Nil: +50 if 0 tricks, -50 if any tricks
        - Blind nil: +100 if 0 tricks, -100 if any tricks

        Overtrick rule: no bag accumulation / sandbagging. Instead, each
        overtrick directly costs 9 points (contract*10 + overtrick*(-9)).
        This strongly incentivizes precise bidding.
        """
        teams = state.teams  # [0, 1, 0, 1]
        team_scores = [0.0, 0.0]

        for team_id in range(2):
            members = [i for i in range(4) if teams[i] == team_id]
            team_bid = 0
            team_tricks = sum(state.tricks_won[i] for i in members)
            score = 0.0

            for pid in members:
                player_bid = state.max_bid[pid]

                if player_bid == "blind_nil":
                    score += 100.0 if state.tricks_won[pid] == 0 else -100.0
                elif player_bid == "nil":
                    score += 50.0 if state.tricks_won[pid] == 0 else -50.0
                else:
                    # Extract numeric bid
                    if isinstance(player_bid, str) and player_bid.startswith("bid_"):
                        team_bid += int(player_bid.split("_")[1])
                    elif isinstance(player_bid, int):
                        team_bid += player_bid

            # Team contract scoring
            if team_bid > 0:
                if team_tricks >= team_bid:
                    overtricks = team_tricks - team_bid
                    score += team_bid * 10 - overtricks * 9
                else:
                    score -= team_bid * 10

            team_scores[team_id] = score

        # Per-player payoffs: own team score - opponent team score
        payoffs = [0.0] * 4
        for pid in range(4):
            own_team = teams[pid]
            opp_team = 1 - own_team
            payoffs[pid] = team_scores[own_team] - team_scores[opp_team]

        return payoffs
