from __future__ import annotations

import pytest

from trick_taking.card import Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.game_state import GameState, Phase
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


_CARDS_BY_ID = {card.card_id: card for card in _STANDARD_CARDS}


def _late_state(
    hands: list[list[int]],
    *,
    turn: int,
    tricks_won: list[int],
    bids: list[str],
) -> GameState:
    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = [
        [_CARDS_BY_ID[card_id] for card_id in hand]
        for hand in hands
    ]
    state.hand_bitsets = [
        cards_to_bitset(hand) for hand in state.hands
    ]
    state.all_cards = list(_STANDARD_CARDS)
    state.table_cards = []
    state.turn = turn
    state.trick_leader = turn
    state.trump_suit = Suit.SPADES
    state.spades_broken = True
    state.trump_broken = True
    state.tricks_played = 10
    state.tricks_won = tricks_won
    state.max_bid = bids
    state.teams = [0, 1, 0, 1]
    return state


def test_quick_trick_pruning_regression_returns_exact_q_values() -> None:
    state = _late_state(
        [
            [20, 30, 12],
            [3, 19, 13],
            [16, 49, 51],
            [10, 31, 47],
        ],
        turn=0,
        tricks_won=[2, 1, 3, 4],
        bids=["bid_4", "bid_5", "bid_2", "bid_5"],
    )

    result = ExactDoubleDummyCppFastestSolver().solve_with_q_fast(state)

    assert result == pytest.approx(
        {12: 160.0, 20: 142.0, 30: 151.0}
    )


def test_tt_bound_flag_regression_does_not_cache_bound_as_exact() -> None:
    state = _late_state(
        [
            [8, 18, 23],
            [36, 35, 31],
            [5, 49, 39],
            [38, 13, 27],
        ],
        turn=3,
        tricks_won=[2, 1, 2, 5],
        bids=["bid_1", "bid_3", "bid_3", "bid_3"],
    )

    result = ExactDoubleDummyCppFastestSolver().solve_with_q_fast(state)

    assert result == pytest.approx(
        {13: -47.0, 27: -11.0, 38: -29.0}
    )
