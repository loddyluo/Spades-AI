"""Rule-based following logic for the Spades AI."""
from __future__ import annotations

from spades_ai.game.card import Card, Suit
from spades_ai.game.state import GameState
from spades_ai.game.trick import _beats
from spades_ai.players.rule_based_v2.helpers import get_suit_cards
from spades_ai.players.rule_based_v2.strategy import assess_strategy


def choose_follow(
    state: GameState,
    player: int,
    legal: frozenset[Card],
) -> Card:
    """Choose a card to play when following (not leading) a trick."""
    led_suit = state.current_trick_cards[0].card.suit
    in_suit = frozenset(c for c in legal if c.suit == led_suit)
    priorities = assess_strategy(state, player)

    if in_suit:
        return _follow_in_suit(state, player, in_suit, legal, led_suit, priorities)
    return _follow_off_suit(state, player, legal, led_suit, priorities)


# ---------------------------------------------------------------------------
# In-suit following
# ---------------------------------------------------------------------------

def _follow_in_suit(
    state: GameState,
    player: int,
    in_suit: frozenset[Card],
    legal: frozenset[Card],
    led_suit: Suit,
    priorities: list[str],
) -> Card:
    """Choose card when player has cards in the led suit."""
    sorted_asc = sorted(in_suit, key=lambda c: c.rank)
    sorted_desc = sorted(in_suit, key=lambda c: c.rank, reverse=True)

    # Determine the current winning card in the trick
    winning_card = _current_winner_card(state, led_suit)

    for priority in priorities:
        if priority == "AVOID_OVERTRICKS":
            # Play the smallest card
            return sorted_asc[0]

        if priority == "PROTECT_PARTNER_NIL":
            # Cover partner's nil: play card to beat current winner if needed
            partner = (player + 2) % 4
            # Check if partner is in the trick and at risk
            partner_card = _player_card_in_trick(state, partner)
            if partner_card is not None and winning_card is not None:
                # Partner is winning — protect by playing small
                if _beats(partner_card, winning_card, led_suit):
                    return sorted_asc[0]
            # Otherwise play under the current winner to protect partner
            return sorted_asc[0]

        if priority == "ATTACK_OPP_NIL":
            # Try to play under the nil player's card
            nil_player = _find_nil_player(state, player)
            if nil_player is not None:
                nil_card = _player_card_in_trick(state, nil_player)
                if nil_card is not None:
                    # Play just under nil card if possible
                    under = [c for c in sorted_asc if c.rank < nil_card.rank]
                    if under:
                        return under[-1]  # Highest card under nil card
            return sorted_asc[0]

        if priority in ("NEED_TRICKS", "TEAM_SET"):
            # Play cheapest winner if we can beat current card, else smallest
            if winning_card is not None:
                winners = [c for c in sorted_asc if _beats(c, winning_card, led_suit)]
                if winners:
                    return winners[0]  # Cheapest winner
            elif sorted_desc:
                # No current winner to beat (we led) — play highest
                return sorted_desc[0]
            return sorted_asc[0]

    # Default: smallest in suit
    return sorted_asc[0]


# ---------------------------------------------------------------------------
# Off-suit following
# ---------------------------------------------------------------------------

def _follow_off_suit(
    state: GameState,
    player: int,
    legal: frozenset[Card],
    led_suit: Suit,
    priorities: list[str],
) -> Card:
    """Choose card when player is void in the led suit."""
    spades = [c for c in legal if c.suit == Suit.SPADES]
    non_spades = [c for c in legal if c.suit != Suit.SPADES]
    only_spades = len(non_spades) == 0

    # Sort helpers
    non_spades_desc = sorted(non_spades, key=lambda c: c.rank, reverse=True)
    spades_asc = sorted(spades, key=lambda c: c.rank)

    for priority in priorities:
        if priority == "AVOID_OVERTRICKS":
            # Discard highest non-spade (dump winners)
            if non_spades_desc:
                return non_spades_desc[0]
            # All spades — play smallest
            return spades_asc[0] if spades_asc else min(legal, key=lambda c: c.rank)

        if priority == "PROTECT_PARTNER_NIL":
            partner = (player + 2) % 4
            partner_card = _player_card_in_trick(state, partner)
            winning_card = _current_winner_card(state, led_suit)
            # If partner is currently winning, don't ruff (protect nil)
            if partner_card is not None and winning_card is not None:
                if _beats(partner_card, winning_card, led_suit):
                    # Partner already winning — discard safely
                    if non_spades_desc:
                        return non_spades_desc[-1]  # Lowest non-spade
                    return spades_asc[0] if spades_asc else min(legal, key=lambda c: c.rank)
            # Ruff to prevent opponent from winning
            if spades_asc:
                return spades_asc[0]
            if non_spades_desc:
                return non_spades_desc[-1]
            return min(legal, key=lambda c: c.rank)

        if priority == "ATTACK_OPP_NIL":
            nil_player = _find_nil_player(state, player)
            if nil_player is not None:
                nil_card = _player_card_in_trick(state, nil_player)
                winning_card = _current_winner_card(state, led_suit)
                # If nil player is winning, don't ruff (they take the trick — bad for nil)
                if nil_card is not None and winning_card is not None:
                    if _beats(nil_card, winning_card, led_suit):
                        # Let nil player win — discard safely
                        if non_spades_desc:
                            return non_spades_desc[-1]
                        return spades_asc[0] if spades_asc else min(legal, key=lambda c: c.rank)
            # Ruff to try to win
            if spades_asc:
                return spades_asc[0]
            if non_spades_desc:
                return non_spades_desc[-1]
            return min(legal, key=lambda c: c.rank)

        if priority in ("NEED_TRICKS", "TEAM_SET"):
            winning_card = _current_winner_card(state, led_suit)
            winning_spades = [
                c for c in spades_asc
                if winning_card is not None and _beats(c, winning_card, led_suit)
            ]
            if winning_spades:
                return winning_spades[0]
            if spades_asc:
                if non_spades and _team_need(state, player) >= 4:
                    return min(non_spades, key=lambda c: c.rank)
                return spades_asc[0]
            # No spades — discard lowest non-spade
            if non_spades:
                return min(non_spades, key=lambda c: c.rank)
            return min(legal, key=lambda c: c.rank)

    # Default: discard lowest non-spade or smallest spade
    if non_spades:
        return min(non_spades, key=lambda c: c.rank)
    return min(legal, key=lambda c: c.rank)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _current_winner_card(state: GameState, led_suit: Suit) -> Card | None:
    """Return the card that is currently winning the trick, or None if empty."""
    if not state.current_trick_cards:
        return None
    best_card = state.current_trick_cards[0].card
    for tc in state.current_trick_cards[1:]:
        if _beats(tc.card, best_card, led_suit):
            best_card = tc.card
    return best_card


def _player_card_in_trick(state: GameState, player: int) -> Card | None:
    """Return the card player has played in the current trick, or None."""
    for tc in state.current_trick_cards:
        if tc.player == player:
            return tc.card
    return None


def _find_nil_player(state: GameState, player: int) -> int | None:
    """Find an alive nil bidder among opponents, or None."""
    from spades_ai.game.scoring import BidType
    opp1 = (player + 1) % 4
    opp2 = (player + 3) % 4
    for opp in [opp1, opp2]:
        if (
            state.bids[opp].bid_type in (BidType.NIL, BidType.BLIND_NIL)
            and state.tricks_won[opp] == 0
        ):
            return opp
    return None


def _team_need(state: GameState, player: int) -> int:
    """Return the remaining normal-bid tricks needed by player's team."""
    partner = (player + 2) % 4
    bid = sum(
        state.bids[p].value
        for p in (player, partner)
        if state.bids[p].bid_type.name == "NORMAL"
    )
    tricks = state.tricks_won[player] + state.tricks_won[partner]
    return bid - tricks
