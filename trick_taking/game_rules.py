"""
GameRules abstract base class — the paper's Fig. 2 interface.

Paper reference: Section 4 "General Card Game Play", Fig. 2
"The interface for the domain-specific rules takes the (uni)code corresponding
to a bid, the (uni)code corresponding to the cards, vectors of cards for the
suits, and arrays of integers for game variant and contract chosen."

Each concrete game (Spades, Hearts, Skat, Bridge, etc.) implements this ABC.
The GeneralCardGame driver calls these methods without knowing which game
is being played — achieving the paper's goal of a game-agnostic engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trick_taking.card import Card, RankOrder, Suit
from trick_taking.deck import DeckConfig
from trick_taking.game_state import GameState


class GameRules(ABC):
    """
    Abstract game rules interface (Paper's Fig. 2).

    Each method corresponds to a domain-specific function that the general
    driver loop (Fig. 3) calls. The docstrings reference the paper's naming.

    To implement a new trick-taking game, subclass GameRules and implement
    all abstract methods. The driver loop handles everything else.
    """

    # ─── Identity (paper: "game_name, bid_name, contract_name") ──────

    @property
    @abstractmethod
    def game_name(self) -> str:
        """Text for variant of game played."""
        ...

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players in the game."""
        ...

    @property
    @abstractmethod
    def deck_config(self) -> DeckConfig:
        """Deck specification for this game."""
        ...

    @property
    @abstractmethod
    def cards_per_hand(self) -> int:
        """Number of cards dealt to each player (⌊n/p⌋)."""
        ...

    @property
    def dog_size(self) -> int:
        """Extra cards not dealt to players (Skat=2, Tarot=6, else 0)."""
        return 0

    # ─── Card ordering (paper: rank permutation π) ───────────────────

    @abstractmethod
    def rank_order(self) -> RankOrder:
        """The game's rank permutation π for comparing card strength."""
        ...

    # ─── Trump (paper: "trump_mask") ─────────────────────────────────

    @abstractmethod
    def trump_mask(self, state: GameState) -> set[Suit] | None:
        """
        Return the set of trump suits, or None if no trump.
        Paper: "the subset of trump cards for the game"
        Can depend on state (e.g., trump determined by bidding in Skat).
        """
        ...

    # ─── Teams (paper: "set_team") ───────────────────────────────────

    @abstractmethod
    def set_team(self, state: GameState) -> list[int]:
        """
        Assign team memberships after bidding.
        Paper: "setting a value in the team vector"
        Returns list of team ids, one per player (0 or 1 for two teams).
        """
        ...

    # ─── Scoring (paper: "points_card", "points_trick", "score") ─────

    def initial_points(self, state: GameState) -> list[float]:
        """Paper: "offset score given to the players". Default: all zeros."""
        return [0.0] * state.num_players

    def points_card(self, card: Card) -> int:
        """Paper: "the score value of a card in a game". Default: 0."""
        return 0

    def points_trick(self, trick_cards: list[tuple[int, Card]],
                     state: GameState) -> int:
        """Paper: "the score value of a trick in a game". Default: sum of card points."""
        return sum(self.points_card(card) for _, card in trick_cards)

    @abstractmethod
    def score(self, state: GameState) -> list[float]:
        """
        Paper: "determine the current score of a given player"
        Called at end of game. Returns per-player scores.
        """
        ...

    # ─── Trick winner (paper: "winner_trick") ────────────────────────

    def winner_trick(self, state: GameState) -> int:
        """
        Paper: "determines the winner of a trick"

        Default implementation: highest trump wins, else highest card of
        lead suit. Override for games with special winner rules (e.g.,
        Doppelkopf twin cards).
        """
        table = state.table_cards
        if not table:
            raise ValueError("No cards on table")

        rank_ord = self.rank_order()
        trump_suits = self.trump_mask(state)
        lead_suit = table[0][1].suit

        best_pid = table[0][0]
        best_card = table[0][1]
        best_is_trump = (trump_suits is not None and best_card.suit in trump_suits)

        for pid, card in table[1:]:
            is_trump = (trump_suits is not None and card.suit in trump_suits)

            if is_trump and not best_is_trump:
                # Trump beats non-trump
                best_pid, best_card, best_is_trump = pid, card, True
            elif is_trump and best_is_trump:
                # Both trump — higher rank wins
                if rank_ord.stronger(card.rank, best_card.rank):
                    best_pid, best_card = pid, card
            elif not is_trump and not best_is_trump:
                # Neither trump — must be lead suit and higher rank
                if card.suit == lead_suit and (
                    best_card.suit != lead_suit
                    or rank_ord.stronger(card.rank, best_card.rank)
                ):
                    best_pid, best_card = pid, card
            # else: non-trump vs trump — trump wins, no change

        return best_pid

    # ─── Bidding (paper: "next_bid_turn", "end_bidding") ─────────────

    @property
    def has_bidding(self) -> bool:
        """Whether this game has a bidding phase. Default: False."""
        return False

    def next_bid_turn(self, state: GameState) -> int:
        """Paper: "next player to bid". Default: rotate clockwise."""
        return (state.current_bidder + 1) % state.num_players

    def end_bidding(self, state: GameState) -> bool:
        """Paper: "check if bidding has concluded". Default: True (no bidding)."""
        return True

    def legal_bids(self, state: GameState, player_id: int) -> list[Any]:
        """Legal bid actions for the given player. Default: empty."""
        return []

    # ─── Declaration (paper: "declareGame", "isHandGame", "setGame") ──

    @property
    def has_declaration(self) -> bool:
        """Whether this game has a declaration phase (Skat, Tarot). Default: False."""
        return False

    def legal_declarations(self, state: GameState,
                           player_id: int) -> list[Any]:
        """Legal game declarations. Default: empty."""
        return []

    # ─── Dog/Skat/Talon exchange ─────────────────────────────────────

    @property
    def has_exchange(self) -> bool:
        """Whether this game has a dog/skat/talon exchange. Default: False."""
        return False

    def legal_exchanges(self, state: GameState,
                        player_id: int) -> list[Any]:
        """Legal exchange actions. Default: empty."""
        return []

    # ─── Legal plays (paper: "playable") ─────────────────────────────

    @abstractmethod
    def playable(self, state: GameState, hand: list[Card],
                 player_id: int) -> list[Card]:
        """
        Paper: "given a hand and table cards, determine the reduced
        set of valid cards"

        Core of trick-taking rules: follow suit, trump rules, etc.
        """
        ...

    # ─── End condition (paper: "end_trickgame") ──────────────────────

    def end_trickgame(self, state: GameState) -> bool:
        """
        Paper: "check if game is already over"
        Default: game ends when all hands are empty.
        """
        return all(len(h) == 0 for h in state.hands)
