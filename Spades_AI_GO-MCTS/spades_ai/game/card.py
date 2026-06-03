from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        return ["♣", "♦", "♥", "♠"][self.value]

    def __str__(self) -> str:
        return self.symbol


class Rank(IntEnum):
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

    def __str__(self) -> str:
        names = {11: "J", 12: "Q", 13: "K", 14: "A"}
        return names.get(self.value, str(self.value))


@total_ordering
@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def index(self) -> int:
        return self.suit.value * 13 + (self.rank.value - 2)

    @classmethod
    def from_index(cls, index: int) -> Card:
        suit = Suit(index // 13)
        rank = Rank(index % 13 + 2)
        return cls(rank, suit)

    @classmethod
    def all_cards(cls) -> list[Card]:
        return [cls.from_index(i) for i in range(52)]

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank!r}, {self.suit!r})"

    def __lt__(self, other: Card) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return (self.suit.value, self.rank.value) < (other.suit.value, other.rank.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
