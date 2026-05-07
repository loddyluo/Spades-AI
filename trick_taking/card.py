"""
Card encoding for trick-taking card games.

Paper reference: Section 3 "Data Structures"
- Card Encoding: bitset-based representation for fast constant-time operations
- Card Ordering: rank permutation π = (π₁,...,πₙ) for configurable card strength
- Equivalent cards: adjacent cards in π with same suit are indistinguishable
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Sequence


@unique
class Suit(IntEnum):
    """Card suits. Order follows standard convention (♠ ♥ ♦ ♣)."""
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3

    @property
    def symbol(self) -> str:
        return _SUIT_SYMBOLS[self.value]

    @property
    def short(self) -> str:
        """Single-character abbreviation: S, H, D, C."""
        return self.name[0]

    @classmethod
    def from_short(cls, s: str) -> Suit:
        return _SHORT_TO_SUIT[s.upper()]

_SUIT_SYMBOLS = {0: "♠", 1: "♥", 2: "♦", 3: "♣"}
_SHORT_TO_SUIT: dict[str, Suit] = {s.short: s for s in Suit}


@unique
class Rank(IntEnum):
    """Card ranks. IntEnum value doubles as a natural strength ordering."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def short(self) -> str:
        """Single-character abbreviation: 2-9, T, J, Q, K, A."""
        return _RANK_SHORT[self.value]

    @classmethod
    def from_short(cls, s: str) -> Rank:
        return _SHORT_TO_RANK[s.upper()]

_RANK_SHORT: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
_SHORT_TO_RANK: dict[str, Rank] = {v: Rank(k) for k, v in _RANK_SHORT.items()}


@dataclass(frozen=True, slots=True)
class Card:
    """
    Immutable playing card with suit and rank.

    The `card_id` property provides a unique index 0..51 for bitset encoding
    (paper Section 3: "the card bitset encoding, which provides fast
    constant-time operations in the search algorithms").
    """
    suit: Suit
    rank: Rank

    @property
    def card_id(self) -> int:
        """Unique integer 0..51 for standard deck: suit * 13 + (rank - 2)."""
        return self.suit.value * 13 + (self.rank.value - 2)

    @property
    def bit(self) -> int:
        """Single-bit mask for bitset operations: 1 << card_id."""
        return 1 << self.card_id

    def __str__(self) -> str:
        return f"{self.rank.short}{self.suit.symbol}"

    def __repr__(self) -> str:
        return f"Card({self.suit.short}{self.rank.short})"

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse 'SA', 'H2', etc. (suit + rank format)."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s!r}")
        return cls(Suit.from_short(s[0]), Rank.from_short(s[1]))


class RankOrder:
    """
    Paper's π permutation — configurable rank strength ordering.

    "We also derive the permutation π = (π₁,...,πₙ) of the cards as a
    total order for the card's rank, so that a_{π₁} > a_{π₂} ... > a_{πₙ}."

    Different games use different orderings:
    - Standard: 2 < 3 < ... < K < A
    - Euchre:   9 < T < J < Q < K < A (with special Jack rules)
    - Skat:     7 < 8 < 9 < T < J < Q < K < A (Jack is trump)

    Args:
        ascending_ranks: Ranks in ascending strength order.
    """
    __slots__ = ("_strength",)

    def __init__(self, ascending_ranks: Sequence[Rank]) -> None:
        self._strength: dict[Rank, int] = {
            r: i for i, r in enumerate(ascending_ranks)
        }

    def strength(self, rank: Rank) -> int:
        """Return the strength value for a rank. Higher = stronger."""
        return self._strength[rank]

    def compare(self, a: Rank, b: Rank) -> int:
        """Compare two ranks. Positive if a > b, negative if a < b, 0 if equal."""
        return self._strength[a] - self._strength[b]

    def stronger(self, a: Rank, b: Rank) -> bool:
        """Return True if rank a is strictly stronger than rank b."""
        return self._strength[a] > self._strength[b]


# Pre-built standard rank order: 2 < 3 < ... < K < A
STANDARD_RANK_ORDER = RankOrder(list(Rank))


# ─── Bitset utilities (Paper Section 3: "card bitset encoding") ───────────


def cards_to_bitset(cards: Sequence[Card]) -> int:
    """Convert a collection of cards to a single integer bitset.

    Each card is represented by a single bit at position card.card_id.
    Set operations become bitwise: union = |, intersection = &, difference = & ~.
    """
    bits = 0
    for card in cards:
        bits |= card.bit
    return bits


def bitset_to_cards(bitset: int, all_cards: Sequence[Card] | None = None) -> list[Card]:
    """Convert a bitset back to a list of Card objects.

    Args:
        bitset: Integer bitset where bit i represents card with card_id i.
        all_cards: Card lookup table. If None, builds standard 52-card table.
    """
    if all_cards is None:
        all_cards = _STANDARD_CARDS
    result: list[Card] = []
    for card in all_cards:
        if bitset & card.bit:
            result.append(card)
    return result


def bitset_count(bitset: int) -> int:
    """Count the number of cards in a bitset (popcount)."""
    return bin(bitset).count("1")


def suit_mask(suit: Suit) -> int:
    """Return a bitset with all 13 ranks of the given suit set."""
    base = suit.value * 13
    return ((1 << 13) - 1) << base


# Pre-built standard 52-card lookup table
_STANDARD_CARDS: tuple[Card, ...] = tuple(
    Card(s, r) for s in Suit for r in Rank
)
