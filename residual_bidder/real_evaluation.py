"""Paired evaluation with the unchanged full card-play player."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.random_tape import BidSamplingKey
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.game_state import GameState
from trick_taking.games.spades import SpadesRules


class ActingBidOverride:
    """Override only ``place_bid`` while delegating all card play unchanged."""

    def __init__(
        self,
        inner: Any,
        policy: Any,
        *,
        policy_seed: int,
        deal_id: str,
        room_id: str,
    ) -> None:
        if not callable(getattr(policy, "sample", None)):
            raise TypeError("policy must provide sample")
        if type(policy_seed) is not int:
            raise TypeError("policy_seed must be an integer")
        if not isinstance(deal_id, str) or not deal_id:
            raise ValueError("deal_id must be a nonempty string")
        if not isinstance(room_id, str) or not room_id:
            raise ValueError("room_id must be a nonempty string")
        self._inner = inner
        self._policy = policy
        self._policy_seed = policy_seed
        self._deal_id = deal_id
        self._room_id = room_id
        self._position = -1
        self.last_bid_info: dict[str, Any] | None = None

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def acting_policy(self) -> Any:
        return self._policy

    @property
    def room_id(self) -> str:
        return self._room_id

    def start_game(self, position: int, hand: list[Any], num_players: int) -> None:
        self._position = position
        self._inner.start_game(position, hand, num_players)

    def place_bid(self, legal_bids: list[Any], state_view: dict[str, Any]) -> Any:
        state = state_view.get("state")
        if not isinstance(state, GameState):
            raise ValueError("acting bid override requires state_view['state']")
        if state.current_bidder != self._position:
            raise ValueError("acting bid override was called for the wrong seat")
        bid_index = len(state.bids)
        decision = self._policy.sample(
            state,
            legal_bids,
            BidSamplingKey(
                policy_seed=self._policy_seed,
                deal_id=self._deal_id,
                room_id=self._room_id,
                logical_seat=self._position,
                bid_index=bid_index,
            ),
            strict=True,
        )
        action = getattr(decision, "action", None)
        if not isinstance(action, BidAction):
            raise TypeError("acting policy must return a canonical BidAction")
        value = to_local_bid(action)
        if value not in legal_bids:
            raise ValueError(f"acting policy returned illegal bid {value!r}")
        self.last_bid_info = {
            "chosen_bid": value,
            "policy_id": getattr(decision, "effective_policy_id", None),
            "legal_bids": list(legal_bids),
        }
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass(frozen=True)
class RealDealResult:
    deal_id: str
    shuffle_seed: int
    room_team0_margin: float
    room_team1_margin: float
    duplicate_margin: float


@dataclass(frozen=True)
class RealEvaluationSummary:
    deals: int
    mean_duplicate_margin: float
    standard_error: float
    wins: int
    ties: int
    losses: int
    results: tuple[RealDealResult, ...]


def _play_room(
    shuffle_seed: int,
    candidate: Any,
    opponent: Any,
    player_factory: Callable[[], Any],
    *,
    candidate_team: int,
    policy_seed: int,
    deal_id: str,
    runner_factory: Callable[..., Any],
) -> float:
    room_id = f"candidate-team-{candidate_team}"
    players = [
        ActingBidOverride(
            player_factory(),
            candidate if seat % 2 == candidate_team else opponent,
            policy_seed=policy_seed,
            deal_id=deal_id,
            room_id=room_id,
        )
        for seat in range(4)
    ]
    runner = runner_factory(
        players=players,
        seed=shuffle_seed,
        verbose=False,
        rules=SpadesRules(enable_nil=True, enable_blind_nil=False),
    )
    result = runner.play_game()
    scores = getattr(result, "scores", None)
    if not isinstance(scores, Sequence) or len(scores) != 4:
        raise ValueError("full-play runner must return four player scores")
    margin = float(scores[candidate_team])
    if not math.isfinite(margin):
        raise ValueError("full-play runner returned a non-finite score")
    return margin


def evaluate_real_deal(
    shuffle_seed: int,
    candidate: Any,
    opponent: Any,
    player_factory: Callable[[], Any],
    *,
    policy_seed: int,
    deal_id: str | None = None,
    runner_factory: Callable[..., Any] = SpadesMatchRunner,
) -> RealDealResult:
    """Play both rooms with identical card-play players and swapped bidders."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if type(policy_seed) is not int:
        raise TypeError("policy_seed must be an integer")
    if not callable(player_factory):
        raise TypeError("player_factory must be callable")
    resolved_deal_id = deal_id or f"deal-{shuffle_seed}"
    if not isinstance(resolved_deal_id, str) or not resolved_deal_id:
        raise ValueError("deal_id must be a nonempty string")

    room_team0_margin = _play_room(
        shuffle_seed,
        candidate,
        opponent,
        player_factory,
        candidate_team=0,
        policy_seed=policy_seed,
        deal_id=resolved_deal_id,
        runner_factory=runner_factory,
    )
    room_team1_margin = _play_room(
        shuffle_seed,
        candidate,
        opponent,
        player_factory,
        candidate_team=1,
        policy_seed=policy_seed,
        deal_id=resolved_deal_id,
        runner_factory=runner_factory,
    )
    return RealDealResult(
        deal_id=resolved_deal_id,
        shuffle_seed=shuffle_seed,
        room_team0_margin=room_team0_margin,
        room_team1_margin=room_team1_margin,
        duplicate_margin=room_team0_margin + room_team1_margin,
    )


def evaluate_real_duplicates(
    shuffle_seeds: Sequence[int],
    candidate: Any,
    opponent: Any,
    player_factory: Callable[[], Any],
    *,
    policy_seed: int,
) -> RealEvaluationSummary:
    if not isinstance(shuffle_seeds, Sequence) or not shuffle_seeds:
        raise ValueError("shuffle_seeds must be a nonempty sequence")
    seeds = list(shuffle_seeds)
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("all shuffle seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("shuffle seeds must be unique")
    results = tuple(
        evaluate_real_deal(
            seed,
            candidate,
            opponent,
            player_factory,
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
    return RealEvaluationSummary(
        deals=len(results),
        mean_duplicate_margin=statistics.fmean(margins),
        standard_error=standard_error,
        wins=sum(margin > 0 for margin in margins),
        ties=sum(margin == 0 for margin in margins),
        losses=sum(margin < 0 for margin in margins),
        results=results,
    )
