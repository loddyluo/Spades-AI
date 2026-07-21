from __future__ import annotations

import pytest

from strategy.rule_exact_first4_nil_player import RuleExactFirst4NilPlayer
from trick_taking.card import Card, _STANDARD_CARDS
from trick_taking.game_state import GameState, Phase


class _FailIfDecisionLogicRuns(RuleExactFirst4NilPlayer):
    def _exact_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        raise AssertionError("exact search must not run for a forced action")

    def _rule_play(self, legal_cards: list[Card], state_view: dict) -> Card:
        raise AssertionError("rule logic must not run for a forced action")


@pytest.mark.parametrize("cards_per_hand", [1, 10])
def test_single_legal_card_skips_all_decision_logic(cards_per_hand: int) -> None:
    hands = [
        list(_STANDARD_CARDS[seat * cards_per_hand : (seat + 1) * cards_per_hand])
        for seat in range(4)
    ]
    state = GameState()
    state.phase = Phase.PLAYING
    state.hands = hands
    state.turn = 0
    legal_cards = [hands[0][0]]
    player = _FailIfDecisionLogicRuns(exact_solver=object(), exact_threshold=36)

    chosen = player.play_card(legal_cards, {"state": state})

    assert chosen == legal_cards[0]
    assert player.last_play_info == {"mode": "single_action_direct"}
