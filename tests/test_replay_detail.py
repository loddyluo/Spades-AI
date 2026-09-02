from __future__ import annotations

import pytest

from gui.backend import RuleExactProvider
from gui.game_server import _deal_hands_frontend_compat
from gui.replay_detail import (
    analyze_replay_position,
    build_replay_solver_state,
    serialize_ai_play_info,
)
from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import (
    expand_equivalent_root_q_values,
)


def _complete_replay(seed: int = 2026082101):
    rules = SpadesRules()
    initial_hands = _deal_hands_frontend_compat(seed)
    state = GameState()
    state.init_for_deal(
        4,
        [list(hand) for hand in initial_hands],
        [],
        list(_STANDARD_CARDS),
    )
    state.phase = Phase.PLAYING
    state.teams = [0, 1, 0, 1]
    state.max_bid = ["bid_3", "bid_4", "bid_3", "bid_3"]
    state.bids = [
        Bid(player_id=seat, value=value, is_pass=False)
        for seat, value in enumerate(state.max_bid)
    ]
    state.trump_suit = Suit.SPADES
    state.turn = state.trick_leader = 0
    plays: list[tuple[int, Card]] = []

    while state.tricks_played < 13:
        seat = state.turn
        legal = rules.playable(state, state.hands[seat], seat)
        card = min(legal, key=lambda candidate: candidate.card_id)
        plays.append((seat, card))
        state.play_card_to_table(seat, card)
        if card.suit == Suit.SPADES:
            state.trump_broken = state.spades_broken = True
        state.turn = (seat + 1) % 4
        if state.trick_complete:
            winner = rules.winner_trick(state)
            state.complete_trick(winner)
            state.turn = state.trick_leader = winner

    return initial_hands, state.max_bid, plays


def test_replay_state_rebuilds_immediately_before_target_action() -> None:
    hands, bids, plays = _complete_replay()

    state, target = build_replay_solver_state(
        hands,
        bids,
        plays,
        first_leader=plays[0][0],
        play_index=12,
    )

    assert state.tricks_played == 3
    assert state.table_cards == []
    assert target == plays[12]
    assert state.turn == target[0]
    assert sum(len(hand) for hand in state.hands) == 40


def test_replay_analysis_is_limited_to_last_ten_tricks() -> None:
    hands, bids, plays = _complete_replay()

    with pytest.raises(ValueError, match="后十墩"):
        build_replay_solver_state(
            hands,
            bids,
            plays,
            first_leader=plays[0][0],
            play_index=11,
        )


def test_replay_analysis_returns_q_for_every_legal_action() -> None:
    hands, bids, plays = _complete_replay()

    class CardIdSolver:
        def __init__(self) -> None:
            self.calls = 0

        def solve_with_q_fast(self, state):
            self.calls += 1
            legal = SpadesRules().playable(
                state,
                state.hands[state.turn],
                state.turn,
            )
            return {card.card_id: float(card.card_id) for card in legal}

    result = analyze_replay_position(
        CardIdSolver(),
        hands,
        bids,
        plays,
        first_leader=plays[0][0],
        play_index=12,
    )
    rebuilt, _ = build_replay_solver_state(
        hands,
        bids,
        plays,
        first_leader=plays[0][0],
        play_index=12,
    )
    legal_count = len(
        SpadesRules().playable(
            rebuilt,
            rebuilt.hands[rebuilt.turn],
            rebuilt.turn,
        )
    )

    assert result["trick_number"] == 4
    assert result["played_card"] == (
        f"{plays[12][1].rank.short}{plays[12][1].suit.short}"
    )
    assert len(result["action_q_values"]) == legal_count
    assert sum(row["is_played"] for row in result["action_q_values"]) == 1


def test_http_provider_accepts_portable_replay_payload() -> None:
    hands, _bids, plays = _complete_replay()

    class CardIdSolver:
        def __init__(self) -> None:
            self.calls = 0

        def solve_with_q_fast(self, state):
            self.calls += 1
            legal = SpadesRules().playable(
                state,
                state.hands[state.turn],
                state.turn,
            )
            return {card.card_id: float(card.card_id) for card in legal}

    solver = CardIdSolver()
    provider = RuleExactProvider.__new__(RuleExactProvider)
    provider.exact_solver = solver
    payload = {
        "playIndex": 12,
        "firstLeader": plays[0][0],
        "bids": [
            {"value": 3, "type": "normal"},
            {"value": 4, "type": "normal"},
            {"value": 3, "type": "normal"},
            {"value": 3, "type": "normal"},
        ],
        "initialHands": [
            [f"{card.rank.short}{card.suit.short}" for card in hand]
            for hand in hands
        ],
        "plays": [
            {"seat": seat, "card": f"{card.rank.short}{card.suit.short}"}
            for seat, card in plays
        ],
    }
    result = provider.analyze_replay(payload)
    cached = provider.analyze_replay(payload)

    assert result["play_index"] == 12
    assert result["trick_number"] == 4
    assert cached == result
    assert solver.calls == 1


def test_equivalent_native_root_actions_inherit_representative_q() -> None:
    low = Card(Suit.HEARTS, Rank.TWO)
    high = Card(Suit.HEARTS, Rank.FOUR)
    state = GameState()
    state.init_for_deal(
        4,
        [
            [low, high],
            [Card(Suit.CLUBS, Rank.TWO)],
            [Card(Suit.DIAMONDS, Rank.TWO)],
            [Card(Suit.SPADES, Rank.TWO)],
        ],
        [],
        [],
    )
    state.phase = Phase.PLAYING
    state.teams = [0, 1, 0, 1]
    state.turn = state.trick_leader = 0
    state.trump_suit = Suit.SPADES

    expanded = expand_equivalent_root_q_values(
        state,
        {high.card_id: 17.0},
        [low, high],
    )

    assert expanded == {low.card_id: 17.0, high.card_id: 17.0}


def test_ai_play_info_serialization_preserves_cards_without_python_objects() -> None:
    chosen = Card(Suit.SPADES, Rank.ACE)
    payload = serialize_ai_play_info(
        {
            "mode": "exact_is_determinized",
            "action_scores": [{"action": chosen, "value": 0.0}],
        },
        chosen_card=chosen,
        seat=2,
    )

    assert payload["schema_version"] == 1
    assert payload["chosen_card"] == "AS"
    assert payload["action_scores"] == [{"action": "AS", "value": 0.0}]
