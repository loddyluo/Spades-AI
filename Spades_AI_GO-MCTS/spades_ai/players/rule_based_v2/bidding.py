"""Rule-based bidding logic for the Spades AI."""
from __future__ import annotations

from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.scoring import BidType
from spades_ai.game.state import Bid

ALL_SUITS = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]


def _get_suit(hand: frozenset[Card], suit: Suit) -> list[Card]:
    return sorted([c for c in hand if c.suit == suit], key=lambda c: c.rank, reverse=True)


def _count_suit(hand: frozenset[Card], suit: Suit) -> int:
    return sum(1 for c in hand if c.suit == suit)


def estimate_tricks(hand: frozenset[Card]) -> float:
    """Estimate the number of tricks this hand is likely to win."""
    tricks = 0.0

    spades = _get_suit(hand, Suit.SPADES)
    spade_count = len(spades)

    for i, card in enumerate(spades):
        if card.rank == Rank.ACE:
            tricks += 1.0
        elif card.rank == Rank.KING:
            tricks += 0.9
        elif card.rank == Rank.QUEEN:
            tricks += 0.7 if spade_count >= 3 else 0.4
        elif card.rank == Rank.JACK:
            tricks += 0.5 if spade_count >= 4 else 0.2
        else:
            if i >= 4 and spade_count >= 5:
                tricks += 0.4

    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
        cards = _get_suit(hand, suit)
        suit_count = len(cards)
        if suit_count == 0:
            # A void is real ruffing potential, but with only a short/medium
            # trump holding it should not push marginal hands into overbids.
            if spade_count >= 4:
                tricks += 0.5
            elif spade_count >= 1:
                tricks += 0.25
            continue
        for card in cards:
            if card.rank == Rank.ACE:
                tricks += 0.95
            elif card.rank == Rank.KING:
                tricks += 0.7 if suit_count >= 2 else 0.3
            elif card.rank == Rank.QUEEN:
                tricks += 0.4 if suit_count >= 3 else 0.1
        if suit_count == 1 and spade_count >= 2:
            tricks += 0.3

    return tricks


def should_bid_nil(hand: frozenset[Card]) -> bool:
    """Return True if the hand qualifies for a nil bid."""
    if any(c.rank >= Rank.QUEEN for c in _get_suit(hand, Suit.SPADES)):
        return False
    if any(c.rank == Rank.ACE for c in hand):
        return False
    non_spade = [c.rank for c in hand if c.suit != Suit.SPADES]
    max_ns = max(non_spade) if non_spade else Rank.TWO
    max_suit = max(_count_suit(hand, s) for s in ALL_SUITS)
    if max_ns <= Rank.TEN and max_suit <= 4:
        return True
    if all(c.rank <= Rank.TEN for c in hand):
        return True
    return False


def rule_based_bid(
    hand: frozenset[Card],
    prev_bids: list[Bid],
    position: int,
) -> Bid:
    """Produce a rule-based bid for the given hand and game context."""
    if should_bid_nil(hand):
        return Bid(value=0, bid_type=BidType.NIL)

    bid = round(estimate_tricks(hand))

    # Bid+1 counterfactuals show that hands with 4+ spades and a moderate
    # team contract often underbid: extra tricks become immediate -9
    # overtricks under our no-bags scoring.  Raise once when the team bid is
    # still modest; avoid already-high team contracts where set risk dominates.
    spade_count = _count_suit(hand, Suit.SPADES)
    high_spade_count = sum(c.rank >= Rank.JACK for c in _get_suit(hand, Suit.SPADES))
    ace_count = sum(c.rank == Rank.ACE for c in hand)
    queen_count = sum(c.rank == Rank.QUEEN for c in hand)

    partner_bid = _get_partner_bid(prev_bids, position)
    if partner_bid is not None:
        if partner_bid.bid_type in (BidType.NIL, BidType.BLIND_NIL):
            bid = max(bid, 4)
        else:
            if partner_bid.value + bid > 10:
                bid = max(1, bid - 1)

    if (
        spade_count >= 4
        and bid <= 4
        and (partner_bid is None or partner_bid.bid_type == BidType.NORMAL)
        and ((partner_bid.value if partner_bid is not None else 0) + bid) <= 6
    ):
        bid += 1

    if (
        spade_count == 3
        and high_spade_count >= 2
        and ace_count == 0
        and (partner_bid is None or partner_bid.bid_type == BidType.NORMAL)
        and ((partner_bid.value if partner_bid is not None else 0) + bid) <= 5
    ):
        bid += 1

    if (
        spade_count >= 6
        and queen_count == 2
        and (partner_bid is None or partner_bid.bid_type == BidType.NORMAL)
    ):
        bid += 1

    # Do not downgrade merely because opponents have bid low.  Under this
    # project's no-bags scoring, underbidding turns otherwise useful tricks
    # into immediate -9 overtricks.  Low opponent bids often mean tricks are
    # available, so keep our own estimate unless partner-contract logic above
    # says the team is already too high.

    return Bid(value=max(1, min(bid, 13)), bid_type=BidType.NORMAL)


def _get_partner_bid(prev_bids: list[Bid], position: int) -> Bid | None:
    partner = (position + 2) % 4
    return prev_bids[partner] if partner < len(prev_bids) else None


def _get_opponent_total_bid(prev_bids: list[Bid], position: int) -> int | None:
    total, count = 0, 0
    for opp in [(position + 1) % 4, (position + 3) % 4]:
        if opp < len(prev_bids):
            if prev_bids[opp].bid_type == BidType.NORMAL:
                total += prev_bids[opp].value
            count += 1
    return total if count > 0 else None
