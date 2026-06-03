"""Rule-based player — integrates bidding and play strategies."""
from __future__ import annotations

from spades_ai.game.card import Card
from spades_ai.game.legal_moves import get_legal_moves
from spades_ai.game.scoring import BidType
from spades_ai.game.state import Bid, GameState
from spades_ai.players.rule_based.bidding import rule_based_bid
from spades_ai.players.rule_based.following import choose_follow
from spades_ai.players.rule_based.leading import choose_lead
from spades_ai.players.rule_based.nil_play import nil_player_follow, nil_player_lead


class RuleBasedPlayer:
    """Deterministic rule-based Spades player."""

    def choose_bid(self, state: GameState) -> Bid:
        player = state.current_player
        return rule_based_bid(state.hands[player], state.bids, player)

    def choose_card(self, state: GameState) -> Card:
        player = state.current_player
        hand = state.hands[player]
        is_leading = len(state.current_trick_cards) == 0
        bid = state.bids[player]

        legal = get_legal_moves(
            hand=hand,
            led_suit=state.led_suit,
            spades_broken=state.spades_broken,
            is_leading=is_leading,
        )

        if bid.bid_type in (BidType.NIL, BidType.BLIND_NIL):
            if is_leading:
                return nil_player_lead(legal, state)
            led_suit = state.current_trick_cards[0].card.suit
            return nil_player_follow(legal, led_suit, state)

        if is_leading:
            return choose_lead(state, player, legal)
        return choose_follow(state, player, legal)
