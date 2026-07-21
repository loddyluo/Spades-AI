"""Fast paired duplicate evaluation using four rule tricks and one DDS solve."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.hybrid import _initial_state, _play_first_four
from residual_bidder.random_tape import BidSamplingKey
from trick_taking.game_state import Bid
from trick_taking.games.spades import SpadesRules


@dataclass(frozen=True)
class FastDealResult:
    deal_id: str
    shuffle_seed: int
    room_team0_margin: float
    room_team1_margin: float
    duplicate_margin: float
    solver_calls: int


@dataclass(frozen=True)
class FastEvaluationSummary:
    deals: int
    solver_calls: int
    mean_duplicate_margin: float
    standard_error: float
    wins: int
    ties: int
    losses: int
    results: tuple[FastDealResult, ...]


def _run_room(
    shuffle_seed: int,
    candidate: Any,
    opponent: Any,
    solver: Any,
    *,
    candidate_team: int,
    room_id: str,
    policy_seed: int,
    deal_id: str,
) -> float:
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state = _initial_state(shuffle_seed, rules)
    for bid_index in range(rules.num_players):
        bidder = state.current_bidder
        policy = candidate if bidder % 2 == candidate_team else opponent
        legal = rules.legal_bids(state, bidder)
        decision = policy.sample(
            state,
            legal,
            BidSamplingKey(
                policy_seed=policy_seed,
                deal_id=deal_id,
                room_id=room_id,
                logical_seat=bidder,
                bid_index=bid_index,
            ),
            strict=True,
        )
        action = getattr(decision, "action", None)
        if not isinstance(action, BidAction):
            raise TypeError("acting policy must return a canonical BidAction")
        value = to_local_bid(action)
        if value not in legal:
            raise ValueError(f"policy returned illegal bid {value!r} for seat {bidder}")
        state.bids.append(Bid(player_id=bidder, value=value, is_pass=False))
        state.max_bid[bidder] = value
        state.current_bidder = rules.next_bid_turn(state)

    state.teams = rules.set_team(state)
    state.points = rules.initial_points(state)
    _play_first_four(state, rules, shuffle_seed)
    team0_margin = float(solver.solve(state))
    if not math.isfinite(team0_margin):
        raise ValueError("terminal solver returned a non-finite value")
    return team0_margin if candidate_team == 0 else -team0_margin


def evaluate_fast_deal(
    shuffle_seed: int,
    candidate: Any,
    opponent: Any,
    solver: Any,
    *,
    policy_seed: int,
    deal_id: str | None = None,
) -> FastDealResult:
    """Evaluate candidate as both partnerships on one duplicate deal."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if type(policy_seed) is not int:
        raise TypeError("policy_seed must be an integer")
    for name, policy in (("candidate", candidate), ("opponent", opponent)):
        if not callable(getattr(policy, "sample", None)):
            raise TypeError(f"{name} must provide sample")
    if not callable(getattr(solver, "solve", None)):
        raise TypeError("solver must provide solve")
    resolved_deal_id = deal_id or f"deal-{shuffle_seed}"
    if not isinstance(resolved_deal_id, str) or not resolved_deal_id:
        raise ValueError("deal_id must be a nonempty string")

    room_team0_margin = _run_room(
        shuffle_seed,
        candidate,
        opponent,
        solver,
        candidate_team=0,
        room_id="candidate-team-0",
        policy_seed=policy_seed,
        deal_id=resolved_deal_id,
    )
    room_team1_margin = _run_room(
        shuffle_seed,
        candidate,
        opponent,
        solver,
        candidate_team=1,
        room_id="candidate-team-1",
        policy_seed=policy_seed,
        deal_id=resolved_deal_id,
    )
    return FastDealResult(
        deal_id=resolved_deal_id,
        shuffle_seed=shuffle_seed,
        room_team0_margin=room_team0_margin,
        room_team1_margin=room_team1_margin,
        duplicate_margin=room_team0_margin + room_team1_margin,
        solver_calls=2,
    )


def evaluate_fast_duplicates(
    shuffle_seeds: Sequence[int],
    candidate: Any,
    opponent: Any,
    solver: Any,
    *,
    policy_seed: int,
) -> FastEvaluationSummary:
    """Aggregate paired deal margins without room-level resampling."""

    if not isinstance(shuffle_seeds, Sequence) or not shuffle_seeds:
        raise ValueError("shuffle_seeds must be a nonempty sequence")
    seeds = list(shuffle_seeds)
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("all shuffle seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("shuffle seeds must be unique")
    results = tuple(
        evaluate_fast_deal(
            seed,
            candidate,
            opponent,
            solver,
            policy_seed=policy_seed,
        )
        for seed in seeds
    )
    margins = [result.duplicate_margin for result in results]
    standard_error = (
        statistics.stdev(margins) / math.sqrt(len(margins))
        if len(margins) > 1
        else 0.0
    )
    return FastEvaluationSummary(
        deals=len(results),
        solver_calls=sum(result.solver_calls for result in results),
        mean_duplicate_margin=statistics.fmean(margins),
        standard_error=standard_error,
        wins=sum(margin > 0 for margin in margins),
        ties=sum(margin == 0 for margin in margins),
        losses=sum(margin < 0 for margin in margins),
        results=results,
    )
