"""Tests for deck module."""

from trick_taking.card import Card, Suit, Rank
from trick_taking.deck import Deck, DeckConfig, STANDARD_52, SKAT_32, EUCHRE_24


class TestDeckConfig:
    def test_standard_52_size(self) -> None:
        assert STANDARD_52.size == 52

    def test_skat_32_size(self) -> None:
        assert SKAT_32.size == 32

    def test_euchre_24_size(self) -> None:
        assert EUCHRE_24.size == 24

    def test_build_cards(self) -> None:
        cards = STANDARD_52.build_cards()
        assert len(cards) == 52
        assert Card(Suit.SPADES, Rank.ACE) in cards
        assert Card(Suit.CLUBS, Rank.TWO) in cards

    def test_skat_no_low_ranks(self) -> None:
        cards = SKAT_32.build_cards()
        assert Card(Suit.SPADES, Rank.TWO) not in cards
        assert Card(Suit.SPADES, Rank.SEVEN) in cards

    def test_exclude(self) -> None:
        config = DeckConfig(exclude=frozenset({Card(Suit.SPADES, Rank.ACE)}))
        cards = config.build_cards()
        assert len(cards) == 51
        assert Card(Suit.SPADES, Rank.ACE) not in cards


class TestDeck:
    def test_deal(self) -> None:
        deck = Deck(STANDARD_52, seed=42)
        hand = deck.deal(13)
        assert len(hand) == 13
        assert deck.remaining == 39

    def test_deal_all(self) -> None:
        deck = Deck(STANDARD_52, seed=42)
        for _ in range(4):
            deck.deal(13)
        assert deck.remaining == 0

    def test_deterministic_seed(self) -> None:
        deck1 = Deck(STANDARD_52, seed=42)
        deck2 = Deck(STANDARD_52, seed=42)
        hand1 = deck1.deal(13)
        hand2 = deck2.deal(13)
        assert hand1 == hand2

    def test_different_seeds(self) -> None:
        deck1 = Deck(STANDARD_52, seed=42)
        deck2 = Deck(STANDARD_52, seed=99)
        hand1 = deck1.deal(13)
        hand2 = deck2.deal(13)
        assert hand1 != hand2  # Extremely unlikely to be equal
