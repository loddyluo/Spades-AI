"""
Configurable deck for trick-taking card games.

Paper reference: Section 2 "Preliminaries" / Table 1
- Different games use different deck sizes: 52 (Bridge/Spades), 32 (Skat),
  24 (Euchre), 78 (Tarot), etc.
- "We used a Mersenne twister to generate random deals for the players."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import random

from trick_taking.card import Card, Suit, Rank


@dataclass(frozen=True)
class DeckConfig:
    """
    Immutable specification of a card deck.

    Different trick-taking games use different decks:
    - Spades/Bridge/Hearts: standard 52 (all 4 suits, ranks 2-A)
    - Skat/Schafkopf: 32 cards (ranks 7-A)
    - Euchre: 24 cards (ranks 9-A)
    - Tarot: 78 cards (standard + 22 trumps)
    - Doppelkopf: 48 cards (2x ranks 9-A, no 2x nines variant)
    """
    suits: tuple[Suit, ...] = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
    min_rank: Rank = Rank.TWO
    max_rank: Rank = Rank.ACE
    exclude: frozenset[Card] = field(default_factory=frozenset)
    duplicates: int = 1  # >1 for Doppelkopf-style double decks

    @property
    def size(self) -> int:
        """Total number of cards in the deck."""
        count = 0
        for s in self.suits:
            for r in Rank:
                if self.min_rank <= r <= self.max_rank:
                    if Card(s, r) not in self.exclude:
                        count += 1
        return count * self.duplicates

    def build_cards(self) -> list[Card]:
        """Generate the ordered list of cards for this deck."""
        cards: list[Card] = []
        for _ in range(self.duplicates):
            for s in self.suits:
                for r in Rank:
                    if self.min_rank <= r <= self.max_rank:
                        c = Card(s, r)
                        if c not in self.exclude:
                            cards.append(c)
        return cards


# ─── Pre-built deck configurations (Paper Table 1) ───────────────────────

STANDARD_52 = DeckConfig()
SKAT_32 = DeckConfig(min_rank=Rank.SEVEN)
EUCHRE_24 = DeckConfig(min_rank=Rank.NINE)
BELOTE_32 = DeckConfig(min_rank=Rank.SEVEN)
DOPPELKOPF_48 = DeckConfig(min_rank=Rank.NINE, duplicates=2)


class Deck:
    """
    Shuffled deck instance. Uses Mersenne Twister for randomness
    as recommended by the paper.

    "We used a Mersenne twister to generate random deals for the players.
    Object inheritance and virtual functions are used to foster a vector
    of different AI or human players."
    """
    __slots__ = ("_cards", "_index", "_rng")

    def __init__(self, config: DeckConfig, seed: Optional[int] = None) -> None:
        self._cards = config.build_cards()
        self._index = 0
        self._rng = random.Random(seed)  # Mersenne Twister
        self._rng.shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        """Deal n cards from the top of the deck."""
        if self._index + n > len(self._cards):
            raise ValueError(
                f"Cannot deal {n} cards, only {self.remaining} remaining"
            )
        dealt = self._cards[self._index : self._index + n]
        self._index += n
        return dealt

    @property
    def remaining(self) -> int:
        """Number of undealt cards."""
        return len(self._cards) - self._index

    @property
    def all_cards(self) -> list[Card]:
        """Full card list (for reference, e.g., knowledge vector initialization)."""
        return list(self._cards)
