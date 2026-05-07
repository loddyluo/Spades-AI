"""
Shared mutable game state for the general driver loop.

Paper reference: Section 4 "General Card Game Play" / Fig. 3
"We also have a function to update the state of the game, the data structures
like the table cards, the bids, the tricks, the points, the turn."

This is the central data structure that the driver (Fig. 3) mutates and that
GameRules queries to make rule decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from trick_taking.card import Card, Suit, cards_to_bitset


class Phase(Enum):
    """Game phases in the driver loop (Fig. 3)."""
    DEALING = auto()
    BIDDING = auto()
    DECLARING = auto()  # declareGame / isHandGame / setGame
    EXCHANGING = auto()  # Dog/Skat/Talon exchange
    PLAYING = auto()
    SCORING = auto()


@dataclass
class Bid:
    """A single bid entry in the bid stack."""
    player_id: int
    value: Any  # int, str, or game-specific bid object
    is_pass: bool = False


@dataclass
class TrickRecord:
    """Record of a completed trick."""
    cards: list[tuple[int, Card]]  # (player_id, card) in order played
    winner: int                     # player_id who won
    leader: int                     # player_id who led

    @property
    def points(self) -> int:
        """Override in scoring to compute trick point value."""
        return 0


@dataclass
class GameState:
    """
    Complete mutable game state, corresponding to the paper's game data structures.

    The driver loop (Fig. 3) maintains this state, and GameRules methods
    inspect it to determine legal actions, winners, etc.

    Key paper data structures mapped here:
    - table_cards: current trick being played (paper's "table")
    - hands/hand_bitsets: per-player cards (bitset for fast ops)
    - bids: bid stack from the bidding phase
    - tricks_won: per-player trick count
    - played_bitset: bitset of all cards played so far
    - teams: team membership (paper's "team : {1,...,p} → {0,1}")
    """
    # ─── Game identity ────────────────────────────────────────────────
    num_players: int = 4
    phase: Phase = Phase.DEALING

    # ─── Dealing ──────────────────────────────────────────────────────
    dealer_seat: int = 0
    hands: list[list[Card]] = field(default_factory=list)
    hand_bitsets: list[int] = field(default_factory=list)  # per-player
    dog: list[Card] = field(default_factory=list)  # Skat/Tarot extra cards
    all_cards: list[Card] = field(default_factory=list)  # full deck reference

    # ─── Bidding (paper: "bids are kept in a stack") ──────────────────
    bids: list[Bid] = field(default_factory=list)
    max_bid: list[Any] = field(default_factory=list)  # per-player max bid
    current_bidder: int = 0

    # ─── Declaration / Contract ───────────────────────────────────────
    declaration: Any = None  # game variant chosen (Skat: Grand, Null, etc.)
    declarer: Optional[int] = None

    # ─── Teams (paper: "team : {1,...,p} → {0,1}") ────────────────────
    teams: list[int] = field(default_factory=list)  # per-player team id

    # ─── Trick play ──────────────────────────────────────────────────
    turn: int = 0  # whose turn it is
    trick_leader: int = 0
    table_cards: list[tuple[int, Card]] = field(default_factory=list)
    trump_suit: Optional[Suit] = None
    trump_broken: bool = False
    spades_broken: bool = False  # Spades-specific alias

    # ─── Scoring ──────────────────────────────────────────────────────
    tricks_won: list[int] = field(default_factory=list)       # per-player count
    cards_won: list[list[Card]] = field(default_factory=list)  # per-player won cards
    points: list[float] = field(default_factory=list)          # per-player points
    trick_history: list[TrickRecord] = field(default_factory=list)

    # ─── Tracking ─────────────────────────────────────────────────────
    played_bitset: int = 0  # bitset of all cards played
    tricks_played: int = 0
    round_number: int = 0  # for multi-round games

    def init_for_deal(self, num_players: int, hands: list[list[Card]],
                      dog: list[Card], all_cards: list[Card]) -> None:
        """Initialize state for a new deal."""
        self.num_players = num_players
        self.hands = hands
        self.hand_bitsets = [cards_to_bitset(h) for h in hands]
        self.dog = dog
        self.all_cards = all_cards
        self.bids = []
        self.max_bid = [None] * num_players
        self.declaration = None
        self.declarer = None
        self.teams = [0] * num_players
        self.table_cards = []
        self.trump_broken = False
        self.spades_broken = False
        self.tricks_won = [0] * num_players
        self.cards_won = [[] for _ in range(num_players)]
        self.points = [0.0] * num_players
        self.trick_history = []
        self.played_bitset = 0
        self.tricks_played = 0

    @property
    def lead_suit(self) -> Optional[Suit]:
        """Suit of the first card in the current trick."""
        if self.table_cards:
            return self.table_cards[0][1].suit
        return None

    @property
    def trick_complete(self) -> bool:
        """Whether the current trick has cards from all players."""
        return len(self.table_cards) >= self.num_players

    def play_card_to_table(self, player_id: int, card: Card) -> None:
        """Add a card to the current trick and update tracking bitsets."""
        self.table_cards.append((player_id, card))
        self.played_bitset |= card.bit
        # Remove from hand
        self.hands[player_id].remove(card)
        self.hand_bitsets[player_id] &= ~card.bit

    def complete_trick(self, winner: int) -> TrickRecord:
        """Record the completed trick, award to winner, reset table."""
        record = TrickRecord(
            cards=list(self.table_cards),
            winner=winner,
            leader=self.trick_leader,
        )
        self.tricks_won[winner] += 1
        for _, card in self.table_cards:
            self.cards_won[winner].append(card)
        self.trick_history.append(record)
        self.tricks_played += 1
        self.table_cards = []
        return record

    def get_player_view(self, player_id: int) -> dict:
        """
        Return an observable state dict from one player's perspective.
        This is what AIPlayer callbacks receive — no hidden information.
        """
        return {
            "player_id": player_id,
            "hand": list(self.hands[player_id]),
            "hand_size": [len(h) for h in self.hands],
            "phase": self.phase,
            "table_cards": list(self.table_cards),
            "lead_suit": self.lead_suit,
            "trump_suit": self.trump_suit,
            "trump_broken": self.trump_broken,
            "bids": [(b.player_id, b.value, b.is_pass) for b in self.bids],
            "tricks_won": list(self.tricks_won),
            "tricks_played": self.tricks_played,
            "teams": list(self.teams),
            "played_bitset": self.played_bitset,
            "dealer_seat": self.dealer_seat,
            "declaration": self.declaration,
            "trick_leader": self.trick_leader,
        }
