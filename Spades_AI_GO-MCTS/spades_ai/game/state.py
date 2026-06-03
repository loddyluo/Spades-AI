"""Immutable game-state representation for Spades.

Every mutation returns a new GameState via dataclasses.replace(); the
original instance is never modified (frozen dataclass throughout).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

from spades_ai.game.card import Card, Suit
from spades_ai.game.scoring import BidType
from spades_ai.game.trick import Trick, TrickCard


class Phase(Enum):
    BIDDING = auto()
    PLAYING = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class Bid:
    value: int
    bid_type: BidType


@dataclass(frozen=True)
class GameState:
    """Full state of a Spades hand-in-progress.

    Attributes
    ----------
    hands:
        One frozenset[Card] per player (index 0-3), shrinks as cards are played.
    bids:
        Accumulated bids in seat order (grows from 0 to 4 during bidding).
    completed_tricks:
        Fully-played Trick objects in chronological order.
    current_trick_cards:
        TrickCards accumulated for the trick currently in progress.
    current_player:
        Seat index (0-3) of the player whose turn it is.
    leader:
        Seat index of the player who led the current trick.
    trick_number:
        0 during bidding; 1-13 during play.
    tricks_won:
        Per-player cumulative tricks taken (4-tuple).
    spades_broken:
        True once a spade has been played as a ruff (not when leading).
    phase:
        Current phase of the hand.
    void_shown:
        Per-player set of suits the player is known to be void in (inferred
        from following with an off-suit card).
    """

    hands: tuple[frozenset[Card], ...]
    bids: tuple[Bid, ...]
    completed_tricks: tuple[Trick, ...]
    current_trick_cards: tuple[TrickCard, ...]
    current_player: int
    leader: int
    trick_number: int
    tricks_won: tuple[int, ...]
    spades_broken: bool
    phase: Phase
    void_shown: tuple[frozenset[Suit], ...]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def new_game(cls, hands: tuple[frozenset[Card], ...]) -> GameState:
        """Create a fresh state ready for bidding."""
        return cls(
            hands=hands,
            bids=(),
            completed_tricks=(),
            current_trick_cards=(),
            current_player=0,
            leader=0,
            trick_number=0,
            tricks_won=(0, 0, 0, 0),
            spades_broken=False,
            phase=Phase.BIDDING,
            void_shown=(frozenset(), frozenset(), frozenset(), frozenset()),
        )

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def place_bid(self, bid: Bid) -> GameState:
        """Record a bid for the current player and advance to the next seat.

        When all 4 players have bid, transitions to PLAYING phase with
        trick_number=1 and current_player reset to seat 0.
        """
        assert self.phase == Phase.BIDDING, (
            f"place_bid called in phase {self.phase}"
        )
        new_bids = self.bids + (bid,)
        next_player = (self.current_player + 1) % 4
        if len(new_bids) == 4:
            return replace(
                self,
                bids=new_bids,
                current_player=0,
                leader=0,
                trick_number=1,
                phase=Phase.PLAYING,
            )
        return replace(self, bids=new_bids, current_player=next_player)

    # ------------------------------------------------------------------
    # Playing
    # ------------------------------------------------------------------

    def play_card(self, card: Card) -> GameState:
        """Remove *card* from current player's hand and advance game state.

        * Updates void tracking when a player follows off-suit.
        * Updates spades_broken when a spade is used as a ruff.
        * On the 4th card of a trick, resolves the trick and either:
          - advances to the next trick (trick_number + 1), or
          - transitions to FINISHED if it was trick 13.
        """
        assert self.phase == Phase.PLAYING, (
            f"play_card called in phase {self.phase}"
        )
        player = self.current_player

        # --- remove card from hand ----------------------------------------
        new_hand = self.hands[player] - {card}
        new_hands: tuple[frozenset[Card], ...] = tuple(
            new_hand if i == player else h for i, h in enumerate(self.hands)
        )

        # --- determine led suit (from first card of current trick) ----------
        led_suit: Suit = (
            self.current_trick_cards[0].card.suit
            if self.current_trick_cards
            else card.suit
        )

        # --- void tracking --------------------------------------------------
        new_void_shown = self.void_shown
        if self.current_trick_cards and card.suit != led_suit:
            # Player could not follow suit → mark them void in led suit
            player_voids: frozenset[Suit] = self.void_shown[player] | {led_suit}
            new_void_shown = tuple(
                player_voids if i == player else v
                for i, v in enumerate(self.void_shown)
            )

        # --- spades broken flag --------------------------------------------
        # Only set when spades used as ruff (not when spades are the led suit)
        new_spades_broken: bool = self.spades_broken or (
            card.suit == Suit.SPADES and led_suit != Suit.SPADES
        )

        # --- accumulate trick card -----------------------------------------
        tc = TrickCard(player=player, card=card)
        new_trick_cards = self.current_trick_cards + (tc,)
        next_player = (player + 1) % 4

        # --- trick complete? -----------------------------------------------
        if len(new_trick_cards) == 4:
            return self._resolve_trick(
                new_hands=new_hands,
                new_trick_cards=new_trick_cards,
                led_suit=led_suit,
                new_spades_broken=new_spades_broken,
                new_void_shown=new_void_shown,
            )

        # --- trick still in progress ---------------------------------------
        return replace(
            self,
            hands=new_hands,
            current_trick_cards=new_trick_cards,
            current_player=next_player,
            spades_broken=new_spades_broken,
            void_shown=new_void_shown,
        )

    def _resolve_trick(
        self,
        new_hands: tuple[frozenset[Card], ...],
        new_trick_cards: tuple[TrickCard, ...],
        led_suit: Suit,
        new_spades_broken: bool,
        new_void_shown: tuple[frozenset[Suit], ...],
    ) -> GameState:
        """Finalise a completed trick and return the updated state."""
        trick = Trick(cards=new_trick_cards, led_suit=led_suit)
        winner = trick.winner()

        new_tricks_won: tuple[int, ...] = tuple(
            t + 1 if i == winner else t for i, t in enumerate(self.tricks_won)
        )
        new_completed = self.completed_tricks + (trick,)

        if self.trick_number == 13:
            # Final trick — transition to FINISHED
            return replace(
                self,
                hands=new_hands,
                completed_tricks=new_completed,
                current_trick_cards=(),
                current_player=winner,
                leader=winner,
                trick_number=self.trick_number,
                tricks_won=new_tricks_won,
                spades_broken=new_spades_broken,
                phase=Phase.FINISHED,
                void_shown=new_void_shown,
            )

        return replace(
            self,
            hands=new_hands,
            completed_tricks=new_completed,
            current_trick_cards=(),
            current_player=winner,
            leader=winner,
            trick_number=self.trick_number + 1,
            tricks_won=new_tricks_won,
            spades_broken=new_spades_broken,
            void_shown=new_void_shown,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def team_tricks(self, team: int) -> int:
        """Return total tricks won by the given team (0 = N/S, 1 = E/W)."""
        if team == 0:
            return self.tricks_won[0] + self.tricks_won[2]
        return self.tricks_won[1] + self.tricks_won[3]

    def get_team(self, player: int) -> int:
        """Return the team index (0 or 1) for *player*."""
        return player % 2

    def get_partner(self, player: int) -> int:
        """Return the seat index of *player*'s partner."""
        return (player + 2) % 4

    @property
    def led_suit(self) -> Suit | None:
        """The suit that was led in the current trick, or None if no card played yet."""
        if self.current_trick_cards:
            return self.current_trick_cards[0].card.suit
        return None
