"""Rule-based leading logic for the Spades AI."""
from __future__ import annotations

from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.state import GameState
from spades_ai.players.rule_based.helpers import (
    get_opponents,
    get_suit_cards,
    is_master,
    opponent_is_void,
)
from spades_ai.players.rule_based.strategy import assess_strategy

_SIDE_SUITS = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]


def choose_lead(
    state: GameState,
    player: int,
    legal: frozenset[Card],
) -> Card:
    """Choose the best card to lead based on current strategic priorities."""
    priorities = assess_strategy(state, player)

    # Dispatch based on highest-priority strategy
    for priority in priorities:
        if priority == "PROTECT_PARTNER_NIL":
            return _lead_to_protect_nil(state, player, legal)
        if priority == "ATTACK_OPP_NIL":
            return _lead_to_attack_nil(state, player, legal)
        if priority == "TEAM_SET":
            return _lead_to_win(state, player, legal)
        if priority == "NEED_TRICKS":
            return _lead_to_win(state, player, legal)
        if priority == "AVOID_OVERTRICKS":
            return _lead_to_lose(state, player, legal)

    # Fallback
    return _lead_to_disrupt(legal)


# ---------------------------------------------------------------------------
# Lead strategies
# ---------------------------------------------------------------------------

def _lead_to_win(state: GameState, player: int, legal: frozenset[Card]) -> Card:
    """Lead a card likely to win a trick."""
    # 1. Master card in a side suit
    for suit in _SIDE_SUITS:
        cards = get_suit_cards(legal, suit)
        for card in cards:
            if is_master(card, suit, state):
                return card

    # 2. Highest card in longest non-spade suit
    best_suit = _longest_suit(legal, _SIDE_SUITS)
    if best_suit is not None:
        cards = get_suit_cards(legal, best_suit)
        if cards:
            return cards[0]  # Highest

    # 3. High spade if spades broken
    if state.spades_broken:
        spades = get_suit_cards(legal, Suit.SPADES)
        if spades:
            return spades[0]

    # 4. Fallback: shortest suit, lowest card
    return _shortest_low(legal)


def _lead_to_lose(state: GameState, player: int, legal: frozenset[Card]) -> Card:
    """Lead a card unlikely to win a trick."""
    opp1, opp2 = get_opponents(player)
    # Smallest non-spade card where opponents might not be void
    non_spades = sorted(
        [c for c in legal if c.suit != Suit.SPADES],
        key=lambda c: c.rank,
    )
    for card in non_spades:
        if not opponent_is_void(card.suit, opp1, state) or not opponent_is_void(card.suit, opp2, state):
            return card
    if non_spades:
        return non_spades[0]
    # Only spades available
    spades = sorted(legal, key=lambda c: c.rank)
    return spades[0]


def _lead_to_protect_nil(state: GameState, player: int, legal: frozenset[Card]) -> Card:
    """Lead a card that protects partner's nil bid."""
    partner = (player + 2) % 4
    partner_voids = state.void_shown[partner]

    # Lead into a suit partner is void in with a high card
    for suit in _SIDE_SUITS:
        if suit in partner_voids:
            cards = get_suit_cards(legal, suit)
            if cards:
                return cards[0]  # Highest

    # Lead high spades (partner can discard safely)
    spades = get_suit_cards(legal, Suit.SPADES)
    if spades:
        return spades[0]

    # Lead any master card
    for suit in _SIDE_SUITS:
        cards = get_suit_cards(legal, suit)
        for card in cards:
            if is_master(card, suit, state):
                return card

    # Fallback
    return _lead_to_win(state, player, legal)


def _lead_to_attack_nil(state: GameState, player: int, legal: frozenset[Card]) -> Card:
    """Lead a card that forces the nil player to win a trick."""
    opp1, opp2 = get_opponents(player)
    # Find which opponent bid nil and is still alive
    from spades_ai.game.scoring import BidType
    nil_player = None
    for opp in [opp1, opp2]:
        if (
            state.bids[opp].bid_type in (BidType.NIL, BidType.BLIND_NIL)
            and state.tricks_won[opp] == 0
        ):
            nil_player = opp
            break

    if nil_player is not None:
        nil_voids = state.void_shown[nil_player]
        # Lead a mid-rank card in a suit the nil player is NOT void in
        for suit in _SIDE_SUITS:
            if suit not in nil_voids:
                cards = get_suit_cards(legal, suit)
                if len(cards) >= 2:
                    # Mid-rank: second from highest
                    return cards[1]
                if cards:
                    return cards[0]

    return _lead_to_win(state, player, legal)


def _lead_to_disrupt(legal: frozenset[Card]) -> Card:
    """Lead the smallest card as a safe discard."""
    return min(legal, key=lambda c: (c.suit.value, c.rank.value))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _longest_suit(legal: frozenset[Card], suits: list[Suit]) -> Suit | None:
    """Return the suit in *suits* with the most cards in *legal*."""
    best: Suit | None = None
    best_count = 0
    for suit in suits:
        count = sum(1 for c in legal if c.suit == suit)
        if count > best_count:
            best_count = count
            best = suit
    return best


def _shortest_low(legal: frozenset[Card]) -> Card:
    """Return the lowest card in the suit with the fewest cards."""
    suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
    best_suit: Suit | None = None
    best_count = 999
    for suit in suits:
        cards = [c for c in legal if c.suit == suit]
        if cards and len(cards) < best_count:
            best_count = len(cards)
            best_suit = suit
    if best_suit is not None:
        return min(
            (c for c in legal if c.suit == best_suit),
            key=lambda c: c.rank,
        )
    return min(legal, key=lambda c: c.rank)
