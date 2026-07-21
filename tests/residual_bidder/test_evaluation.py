from __future__ import annotations

from types import SimpleNamespace

import pytest

from residual_bidder.actions import BidAction, from_local_bid
from residual_bidder.evaluation import evaluate_fast_deal, evaluate_fast_duplicates
from residual_bidder.random_tape import BidSamplingKey
from trick_taking.game_state import GameState, Phase


class _FixedPolicy:
    def __init__(self, action: BidAction, policy_id: str) -> None:
        self.action = action
        self.policy_id = policy_id
        self.keys: list[BidSamplingKey] = []

    def sample(self, state, legal_bids, key, *, strict):
        assert strict is True
        self.keys.append(key)
        return SimpleNamespace(action=self.action)


class _BidDifferenceSolver:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, state: GameState) -> float:
        self.calls += 1
        assert state.phase is Phase.PLAYING
        assert state.tricks_played == 4
        assert state.table_cards == []
        assert tuple(len(hand) for hand in state.hands) == (9, 9, 9, 9)
        bids = [int(from_local_bid(value)) for value in state.max_bid]
        return float((bids[0] + bids[2] - bids[1] - bids[3]) * 100)


def test_fast_deal_swaps_candidate_partnership_and_sums_two_rooms() -> None:
    candidate = _FixedPolicy(BidAction.BID_6, "candidate")
    opponent = _FixedPolicy(BidAction.BID_5, "nsfp")
    solver = _BidDifferenceSolver()

    result = evaluate_fast_deal(
        202607213000,
        candidate,
        opponent,
        solver,
        policy_seed=77,
    )

    assert result.room_team0_margin == 200.0
    assert result.room_team1_margin == 200.0
    assert result.duplicate_margin == 400.0
    assert result.solver_calls == 2
    assert solver.calls == 2
    assert {key.room_id for key in candidate.keys} == {"candidate-team-0", "candidate-team-1"}
    assert {key.logical_seat for key in candidate.keys} == {0, 1, 2, 3}


def test_identical_deterministic_policy_has_exactly_zero_duplicate_margin() -> None:
    policy = _FixedPolicy(BidAction.BID_4, "same")

    result = evaluate_fast_deal(
        202607213001,
        policy,
        policy,
        _BidDifferenceSolver(),
        policy_seed=88,
    )

    assert result.room_team0_margin == -result.room_team1_margin
    assert result.duplicate_margin == 0.0


def test_fast_duplicate_summary_is_reproducible_and_paired() -> None:
    seeds = [202607213010, 202607213011, 202607213012]

    first = evaluate_fast_duplicates(
        seeds,
        _FixedPolicy(BidAction.BID_6, "candidate"),
        _FixedPolicy(BidAction.BID_5, "nsfp"),
        _BidDifferenceSolver(),
        policy_seed=99,
    )
    second = evaluate_fast_duplicates(
        seeds,
        _FixedPolicy(BidAction.BID_6, "candidate"),
        _FixedPolicy(BidAction.BID_5, "nsfp"),
        _BidDifferenceSolver(),
        policy_seed=99,
    )

    assert first == second
    assert first.deals == 3
    assert first.solver_calls == 6
    assert first.mean_duplicate_margin == 400.0
    assert first.standard_error == pytest.approx(0.0)
    assert (first.wins, first.ties, first.losses) == (3, 0, 0)
