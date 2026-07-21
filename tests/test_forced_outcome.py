from __future__ import annotations

import copy

import pytest

from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.forced_outcome import (
    ShowdownStateError,
    apply_showdown_continuation,
    check_for_showdown,
    outcome_signature,
)
from trick_taking.game_state import GameState, Phase, TrickRecord


def _card(code: str) -> Card:
    return Card(Suit.from_short(code[-1]), Rank.from_short(code[:-1]))


_HISTORY = [
    (0, 1, [(0, "3D"), (1, "AD"), (2, "JD"), (3, "QD")]),
    (1, 2, [(1, "6C"), (2, "KC"), (3, "9C"), (0, "JC")]),
    (2, 3, [(2, "8D"), (3, "KD"), (0, "9D"), (1, "2C")]),
    (3, 2, [(3, "TC"), (0, "KH"), (1, "4C"), (2, "AC")]),
    (2, 1, [(2, "2D"), (3, "4D"), (0, "6D"), (1, "5S")]),
    (1, 2, [(1, "8H"), (2, "JH"), (3, "5H"), (0, "9H")]),
    (2, 2, [(2, "8C"), (3, "5C"), (0, "3H"), (1, "7C")]),
    (2, 3, [(2, "7D"), (3, "TS"), (0, "TD"), (1, "AH")]),
    (3, 1, [(3, "2S"), (0, "8S"), (1, "AS"), (2, "JS")]),
    (1, 3, [(1, "6H"), (2, "2H"), (3, "6S"), (0, "QH")]),
    (3, 1, [(3, "QC"), (0, "7S"), (1, "QS"), (2, "3C")]),
]


def _fixed_state() -> GameState:
    hands = [
        [_card("4H"), _card("9S")],
        [_card("TH"), _card("KS")],
        [_card("7H"), _card("5D")],
        [_card("3S"), _card("4S")],
    ]
    history = [
        TrickRecord(
            leader=leader,
            winner=winner,
            cards=[(seat, _card(code)) for seat, code in cards],
        )
        for leader, winner, cards in _HISTORY
    ]
    played = [card for trick in history for _, card in trick.cards]

    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = hands
    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]
    state.all_cards = list(_STANDARD_CARDS)
    state.max_bid = ["nil", "bid_1", "bid_1", "bid_1"]
    state.teams = [0, 1, 0, 1]
    state.turn = 1
    state.trick_leader = 1
    state.table_cards = []
    state.trump_suit = Suit.SPADES
    state.trump_broken = True
    state.spades_broken = True
    state.tricks_won = [0, 4, 4, 3]
    state.cards_won = [
        [],
        [card for trick in history if trick.winner == 1 for _, card in trick.cards],
        [card for trick in history if trick.winner == 2 for _, card in trick.cards],
        [card for trick in history if trick.winner == 3 for _, card in trick.cards],
    ]
    state.trick_history = history
    state.played_bitset = cards_to_bitset(played)
    state.tricks_played = 11
    return state


class _FixedSolver:
    def __init__(self, signature: tuple[int, int] = (4, 0)) -> None:
        self.signature = signature
        self.calls = 0

    def analyze_forced_outcome(self, state, time_budget_seconds=1.0):
        self.calls += 1
        return {
            "status": "fixed",
            "team0_final_tricks": self.signature[0],
            "nil_broken_mask": self.signature[1],
            "nodes_searched": 1,
            "elapsed_ms": 0.1,
        }


@pytest.mark.parametrize("status", ["variable", "timeout"])
def test_non_fixed_native_result_is_forwarded(status: str) -> None:
    class Solver:
        def analyze_forced_outcome(self, state, time_budget_seconds=1.0):
            return {"status": status}

    check = check_for_showdown(_fixed_state(), Solver())

    assert check.status == status
    assert check.resolution is None
    assert check.to_payload() == {"status": status}


def test_fixed_result_builds_and_verifies_complete_legal_continuation() -> None:
    state = _fixed_state()

    check = check_for_showdown(state, _FixedSolver())

    assert check.status == "fixed"
    assert check.resolution is not None
    assert check.resolution.team_tricks == (4, 9)
    assert check.resolution.nil_outcomes == (True, None, None, None)
    assert check.resolution.final_tricks_won == (0, 5, 4, 4)
    assert len(check.resolution.continuation) == 8

    completed = apply_showdown_continuation(
        state,
        check.resolution.continuation,
    )
    assert completed.tricks_played == 13
    assert completed.table_cards == []
    assert completed.tricks_won == list(check.resolution.final_tricks_won)
    assert outcome_signature(completed) == (4, 0)
    assert len(completed.trick_history) == 13

    payload = check.to_payload()
    assert payload["resolution"]["teamTricks"] == [4, 9]
    assert payload["resolution"]["nilOutcomes"] == [True, None, None, None]
    assert len(payload["resolution"]["continuation"]) == 8
    assert set(payload["resolution"]["continuation"][0]) == {"seat", "card"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.table_cards.append((state.turn, state.hands[state.turn][0])),
        lambda state: [hand.append(_card("2S")) for hand in state.hands],
        lambda state: state.hands[1].__setitem__(0, state.hands[0][0]),
        lambda state: state.max_bid.__setitem__(2, None),
        lambda state: setattr(state, "tricks_played", 10),
    ],
    ids=["mid-trick", "six-cards", "duplicate", "incomplete-bid", "history-count"],
)
def test_invalid_authoritative_states_are_rejected_before_native_search(mutate) -> None:
    state = _fixed_state()
    mutate(state)
    solver = _FixedSolver()

    with pytest.raises(ShowdownStateError):
        check_for_showdown(state, solver)

    assert solver.calls == 0


def test_native_and_python_signature_mismatch_suppresses_showdown() -> None:
    with pytest.raises(RuntimeError, match="signature mismatch"):
        check_for_showdown(_fixed_state(), _FixedSolver((5, 0)))


def test_continuation_is_revalidated_before_application() -> None:
    state = _fixed_state()
    resolution = check_for_showdown(state, _FixedSolver()).resolution
    assert resolution is not None
    bad = list(resolution.continuation)
    bad[0] = copy.replace(bad[0], seat=0) if hasattr(copy, "replace") else type(bad[0])(
        seat=0,
        card=bad[0].card,
    )

    with pytest.raises(ValueError, match="seat mismatch"):
        apply_showdown_continuation(state, bad)
