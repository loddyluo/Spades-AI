"""Tests for card module — Card, Suit, Rank, RankOrder, bitset utilities."""

import pytest

from trick_taking.card import (
    Card, Suit, Rank, RankOrder, STANDARD_RANK_ORDER,
    cards_to_bitset, bitset_to_cards, bitset_count, suit_mask,
)


class TestSuit:
    def test_short_names(self) -> None:
        assert Suit.SPADES.short == "S"
        assert Suit.HEARTS.short == "H"
        assert Suit.DIAMONDS.short == "D"
        assert Suit.CLUBS.short == "C"

    def test_from_short(self) -> None:
        assert Suit.from_short("S") == Suit.SPADES
        assert Suit.from_short("h") == Suit.HEARTS

    def test_symbols(self) -> None:
        assert Suit.SPADES.symbol == "♠"
        assert Suit.HEARTS.symbol == "♥"


class TestRank:
    def test_short_names(self) -> None:
        assert Rank.TWO.short == "2"
        assert Rank.TEN.short == "T"
        assert Rank.ACE.short == "A"

    def test_from_short(self) -> None:
        assert Rank.from_short("T") == Rank.TEN
        assert Rank.from_short("a") == Rank.ACE


class TestCard:
    def test_creation(self) -> None:
        card = Card(Suit.SPADES, Rank.ACE)
        assert card.suit == Suit.SPADES
        assert card.rank == Rank.ACE

    def test_frozen(self) -> None:
        card = Card(Suit.SPADES, Rank.ACE)
        with pytest.raises(AttributeError):
            card.suit = Suit.HEARTS  # type: ignore

    def test_card_id(self) -> None:
        # Suit.SPADES=0, Rank.TWO=2 → 0*13 + 0 = 0
        assert Card(Suit.SPADES, Rank.TWO).card_id == 0
        # Suit.SPADES=0, Rank.ACE=14 → 0*13 + 12 = 12
        assert Card(Suit.SPADES, Rank.ACE).card_id == 12
        # Suit.CLUBS=3, Rank.ACE=14 → 3*13 + 12 = 51
        assert Card(Suit.CLUBS, Rank.ACE).card_id == 51

    def test_bit(self) -> None:
        card = Card(Suit.SPADES, Rank.TWO)
        assert card.bit == 1 << 0

    def test_str(self) -> None:
        card = Card(Suit.SPADES, Rank.ACE)
        assert str(card) == "A♠"

    def test_from_str(self) -> None:
        card = Card.from_str("SA")
        assert card == Card(Suit.SPADES, Rank.ACE)
        card2 = Card.from_str("H2")
        assert card2 == Card(Suit.HEARTS, Rank.TWO)

    def test_hashable(self) -> None:
        card1 = Card(Suit.SPADES, Rank.ACE)
        card2 = Card(Suit.SPADES, Rank.ACE)
        assert card1 == card2
        assert hash(card1) == hash(card2)
        s = {card1, card2}
        assert len(s) == 1


class TestRankOrder:
    def test_standard_order(self) -> None:
        ro = STANDARD_RANK_ORDER
        assert ro.stronger(Rank.ACE, Rank.KING)
        assert ro.stronger(Rank.KING, Rank.QUEEN)
        assert not ro.stronger(Rank.TWO, Rank.THREE)

    def test_custom_order(self) -> None:
        # Euchre-style: 9 < T < J < Q < K < A
        euchre_order = RankOrder([
            Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
        ])
        assert euchre_order.stronger(Rank.ACE, Rank.KING)
        assert euchre_order.stronger(Rank.JACK, Rank.TEN)

    def test_compare(self) -> None:
        ro = STANDARD_RANK_ORDER
        assert ro.compare(Rank.ACE, Rank.TWO) > 0
        assert ro.compare(Rank.TWO, Rank.ACE) < 0
        assert ro.compare(Rank.KING, Rank.KING) == 0


class TestBitset:
    def test_cards_to_bitset(self) -> None:
        cards = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.TWO)]
        bs = cards_to_bitset(cards)
        assert bs & Card(Suit.SPADES, Rank.ACE).bit
        assert bs & Card(Suit.HEARTS, Rank.TWO).bit
        assert not (bs & Card(Suit.CLUBS, Rank.THREE).bit)

    def test_roundtrip(self) -> None:
        cards = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.CLUBS, Rank.KING),
        ]
        bs = cards_to_bitset(cards)
        recovered = bitset_to_cards(bs)
        assert set(recovered) == set(cards)

    def test_count(self) -> None:
        cards = [Card(Suit.SPADES, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN]]
        assert bitset_count(cards_to_bitset(cards)) == 3

    def test_suit_mask(self) -> None:
        mask = suit_mask(Suit.SPADES)
        assert bitset_count(mask) == 13
        # All spades should be in the mask
        for r in Rank:
            assert mask & Card(Suit.SPADES, r).bit
        # No hearts
        assert not (mask & Card(Suit.HEARTS, Rank.ACE).bit)
