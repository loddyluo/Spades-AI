"""Four-role paired solver-leaf environment for exactly-one-Nil deals."""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import resource
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.deployment import load_deployed_acting_bidder
from rl.nil_first4_observation import (
    NilFirstFourFeatureEncoderV1,
    build_nil_first_four_observation,
)
from rl.policy_network import PolicyMLP
from rl.solver_leaf_env import (
    CHAMPION_OPPONENT_ID,
    FIRST_TRICKS,
    RULE_OPPONENT_ID,
    TARGET_DIVISOR,
    OpponentPoolConfig,
    assert_solver_leaf_boundary,
    derive_action_seed,
    legal_mask_from_ids,
    select_opponent_id,
    select_policy_action,
)
from strategy.rule_based_first4_nil_player import RuleBasedFirst4NilPlayer
from strategy.spades_match_runner import build_random_state
from trick_taking.card import Suit
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


NIL_SELF = "nil_self"
NIL_PARTNER = "nil_partner"
NIL_UPPER = "nil_upper"
NIL_LOWER = "nil_lower"
NIL_ROLES = (NIL_SELF, NIL_PARTNER, NIL_UPPER, NIL_LOWER)


def role_for_seat(nil_seat: int, seat: int) -> str:
    """Map an absolute seat to the requested role around the Nil bidder."""

    if not 0 <= nil_seat < 4 or not 0 <= seat < 4:
        raise ValueError("seat ids must be in [0, 3]")
    delta = (seat - nil_seat) % 4
    return (NIL_SELF, NIL_LOWER, NIL_PARTNER, NIL_UPPER)[delta]


class ActingBidder(Protocol):
    policy_id: str

    def choose(
        self,
        state: GameState,
        legal_bids: Sequence[Any],
        *,
        logical_seat: int,
        deal_id: str,
        room_id: str,
    ) -> Any: ...


class TerminalSolver(Protocol):
    def solve(self, state: GameState) -> float: ...


@dataclass(frozen=True, slots=True)
class NilLeafTransition:
    feature: np.ndarray
    legal_mask: np.ndarray
    action: int
    old_log_prob: float
    old_entropy: float
    reward: float
    candidate_index: int
    candidate_team: int
    seat: int
    role: str
    play_index: int


@dataclass(frozen=True, slots=True)
class NilRoomResult:
    candidate_team: int
    nil_seat: int
    team0_margin_points: float
    candidate_margin_points: float
    reward: float
    transitions: tuple[NilLeafTransition, ...]
    solver_seconds: float


@dataclass(frozen=True, slots=True)
class NilDuplicateDealResult:
    deal_id: str
    candidate_index: int
    shuffle_seed: int
    nil_seat: int
    opponent_id: str
    room_team0: NilRoomResult
    room_team1: NilRoomResult
    duplicate_margin_points: float
    solver_calls: int

    @property
    def transitions(self) -> tuple[NilLeafTransition, ...]:
        return self.room_team0.transitions + self.room_team1.transitions

    @property
    def transitions_by_role(self) -> dict[str, tuple[NilLeafTransition, ...]]:
        return {
            role: tuple(item for item in self.transitions if item.role == role)
            for role in NIL_ROLES
        }


@dataclass(frozen=True, slots=True)
class NilCandidateOutcome:
    candidate_index: int
    shuffle_seed: int
    nil_count: int
    result: NilDuplicateDealResult | None

    @property
    def accepted(self) -> bool:
        return self.result is not None

    def __post_init__(self) -> None:
        if self.accepted != (self.nil_count == 1):
            raise ValueError("only exactly-one-Nil candidate deals may be accepted")


@dataclass(frozen=True, slots=True)
class NilCollectionBatch:
    deals: tuple[NilDuplicateDealResult, ...]
    start_candidate_index: int
    next_candidate_index: int
    scanned_candidates: int
    nil_count_histogram: dict[int, int]
    elapsed_seconds: float
    worker_peak_rss_bytes: int
    aggregate_worker_peak_rss_bytes: int

    @property
    def transitions(self) -> tuple[NilLeafTransition, ...]:
        return tuple(item for deal in self.deals for item in deal.transitions)

    @property
    def transitions_by_role(self) -> dict[str, tuple[NilLeafTransition, ...]]:
        transitions = self.transitions
        return {
            role: tuple(item for item in transitions if item.role == role)
            for role in NIL_ROLES
        }

    @property
    def solver_calls(self) -> int:
        return sum(deal.solver_calls for deal in self.deals)


def run_production_single_nil_auction(
    shuffle_seed: int,
    bidder: ActingBidder,
    *,
    deal_id: str,
) -> tuple[GameState | None, int]:
    """Run all four deployed bids and accept only exactly one Nil."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if not isinstance(deal_id, str) or not deal_id:
        raise ValueError("deal_id must be nonempty")
    if not callable(getattr(bidder, "choose", None)):
        raise TypeError("bidder must provide choose")

    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state = build_random_state(shuffle_seed)
    nil_count = 0
    for _ in range(rules.num_players):
        seat = state.current_bidder
        legal = rules.legal_bids(state, seat)
        decision = bidder.choose(
            state,
            legal,
            logical_seat=seat,
            deal_id=deal_id,
            room_id="shared-auction",
        )
        if getattr(decision, "fallback_reason", None) is not None:
            raise RuntimeError(
                f"production Residual bidder fallback: {decision.fallback_reason}"
            )
        expected_policy_id = getattr(bidder, "policy_id", None)
        effective_policy_id = getattr(decision, "effective_policy_id", None)
        if expected_policy_id is not None and effective_policy_id != expected_policy_id:
            raise RuntimeError(
                "production Residual bidder changed effective policy: "
                f"expected {expected_policy_id!r}, got {effective_policy_id!r}"
            )
        action = getattr(decision, "action", None)
        if not isinstance(action, BidAction):
            raise TypeError("production Residual bidder returned a non-canonical action")
        value = to_local_bid(action)
        if value not in legal:
            raise ValueError(f"production Residual bidder returned illegal bid {value!r}")
        nil_count += int(action is BidAction.NIL)
        state.bids.append(Bid(player_id=seat, value=value, is_pass=False))
        state.max_bid[seat] = value
        state.current_bidder = rules.next_bid_turn(state)

    if not rules.end_bidding(state) or any(value is None for value in state.max_bid):
        raise AssertionError("production auction did not produce four actual bids")
    if nil_count != 1:
        return None, nil_count
    state.teams = rules.set_team(state)
    state.points = rules.initial_points(state)
    return state, nil_count


def _nil_rule_players(
    state: GameState,
    *,
    candidate_team: int,
    shuffle_seed: int,
) -> dict[int, RuleBasedFirst4NilPlayer]:
    players: dict[int, RuleBasedFirst4NilPlayer] = {}
    for seat in range(state.num_players):
        if state.teams[seat] == candidate_team:
            continue
        player = RuleBasedFirst4NilPlayer(
            bid_seed=(shuffle_seed & 0x7FFFFFFF) + candidate_team * 100 + seat
        )
        player.start_game(seat, list(state.hands[seat]), state.num_players)
        for record in state.bids:
            player.bid_placed(record.player_id, record.value)
        player.set_teams(list(state.teams), list(state.max_bid))
        players[seat] = player
    if len(players) != 2:
        raise AssertionError("one duplicate room must contain exactly two rule seats")
    return players


def play_nil_solver_leaf_room(
    auction_state: GameState,
    actors: Mapping[str, PolicyMLP],
    solver: TerminalSolver,
    encoder: NilFirstFourFeatureEncoderV1,
    *,
    shuffle_seed: int,
    candidate_index: int,
    candidate_team: int,
    run_seed: int,
    deterministic: bool,
    record_transitions: bool,
    opponent_actors: Mapping[str, PolicyMLP] | None = None,
) -> NilRoomResult:
    """Play four blind tricks with one role-specific actor per candidate seat."""

    if candidate_team not in (0, 1):
        raise ValueError("candidate_team must be zero or one")
    if set(actors) != set(NIL_ROLES):
        raise ValueError("actors must contain exactly the four Nil roles")
    state = copy.deepcopy(auction_state)
    nil_seats = [seat for seat, bid in enumerate(state.max_bid) if bid == "nil"]
    if len(nil_seats) != 1:
        raise ValueError("Nil solver room requires exactly one Nil bidder")
    nil_seat = nil_seats[0]
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    if opponent_actors is not None and set(opponent_actors) != set(NIL_ROLES):
        raise ValueError("opponent_actors must contain exactly the four Nil roles")
    rules_by_seat = (
        _nil_rule_players(state, candidate_team=candidate_team, shuffle_seed=shuffle_seed)
        if opponent_actors is None
        else {}
    )
    for actor in actors.values():
        actor.eval()
    if opponent_actors is not None:
        for actor in opponent_actors.values():
            actor.eval()
    pending: list[
        tuple[np.ndarray, np.ndarray, int, float, float, int, str, int]
    ] = []
    play_index = 0

    for _ in range(FIRST_TRICKS):
        if state.table_cards:
            raise AssertionError("table must be empty at trick start")
        for _ in range(rules.num_players):
            seat = state.turn
            legal = rules.playable(state, state.hands[seat], seat)
            if not legal:
                raise AssertionError(f"seat {seat} has no legal card")
            if state.teams[seat] == candidate_team:
                role = role_for_seat(nil_seat, seat)
                observation = build_nil_first_four_observation(state, seat, legal)
                feature = encoder.encode(observation)
                legal_mask = legal_mask_from_ids(observation.legal_card_ids)
                feature_mask = feature[
                    encoder.LEGAL_START : encoder.LEGAL_START + 52
                ].astype(np.bool_)
                if not np.array_equal(legal_mask, feature_mask):
                    raise AssertionError("input legal mask disagrees with action mask")
                action, log_prob, entropy = select_policy_action(
                    actors[role],
                    feature,
                    legal_mask,
                    deterministic=deterministic,
                    sample_seed=derive_action_seed(
                        run_seed, candidate_index, candidate_team, seat, play_index
                    ),
                )
                legal_by_id = {card.card_id: card for card in legal}
                card = legal_by_id.get(action)
                if card is None:
                    raise RuntimeError("masked Nil policy selected an illegal card")
                if record_transitions:
                    pending.append(
                        (
                            feature.copy(),
                            legal_mask.copy(),
                            action,
                            log_prob,
                            entropy,
                            seat,
                            role,
                            play_index,
                        )
                    )
            else:
                if opponent_actors is None:
                    player = rules_by_seat[seat]
                    card = player.play_card(legal, state.get_player_view(seat))
                else:
                    role = role_for_seat(nil_seat, seat)
                    observation = build_nil_first_four_observation(state, seat, legal)
                    feature = encoder.encode(observation)
                    legal_mask = legal_mask_from_ids(observation.legal_card_ids)
                    action, _, _ = select_policy_action(
                        opponent_actors[role],
                        feature,
                        legal_mask,
                        deterministic=True,
                        sample_seed=derive_action_seed(
                            run_seed, candidate_index, candidate_team, seat, play_index
                        ),
                    )
                    legal_by_id = {candidate.card_id: candidate for candidate in legal}
                    card = legal_by_id.get(action)
                    if card is None:
                        raise RuntimeError("masked Nil opponent selected an illegal card")
            if card not in legal:
                raise ValueError(f"seat {seat} returned an illegal card")
            state.play_card_to_table(seat, card)
            if card.suit == Suit.SPADES:
                state.trump_broken = True
                state.spades_broken = True
            for player in rules_by_seat.values():
                player.card_played(seat, card)
            state.turn = (seat + 1) % state.num_players
            play_index += 1
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.turn = winner
        state.trick_leader = winner

    assert_solver_leaf_boundary(state)
    solver_started = time.perf_counter()
    team0_margin = float(solver.solve(state))
    solver_seconds = time.perf_counter() - solver_started
    if not math.isfinite(team0_margin):
        raise ValueError("terminal solver returned a non-finite value")
    candidate_margin = team0_margin if candidate_team == 0 else -team0_margin
    reward = candidate_margin / TARGET_DIVISOR
    transitions = tuple(
        NilLeafTransition(
            feature=feature,
            legal_mask=legal_mask,
            action=action,
            old_log_prob=old_log_prob,
            old_entropy=old_entropy,
            reward=reward,
            candidate_index=candidate_index,
            candidate_team=candidate_team,
            seat=seat,
            role=role,
            play_index=transition_play_index,
        )
        for (
            feature,
            legal_mask,
            action,
            old_log_prob,
            old_entropy,
            seat,
            role,
            transition_play_index,
        ) in pending
    )
    if record_transitions and len(transitions) != 8:
        raise AssertionError("each room must contain eight candidate decisions")
    return NilRoomResult(
        candidate_team=candidate_team,
        nil_seat=nil_seat,
        team0_margin_points=team0_margin,
        candidate_margin_points=candidate_margin,
        reward=reward,
        transitions=transitions,
        solver_seconds=solver_seconds,
    )


def run_nil_duplicate_candidate(
    candidate_index: int,
    shuffle_seed: int,
    actors: Mapping[str, PolicyMLP],
    bidder: ActingBidder,
    solver: TerminalSolver,
    encoder: NilFirstFourFeatureEncoderV1,
    *,
    run_seed: int,
    deterministic: bool,
    record_transitions: bool,
    opponent_pool_config: OpponentPoolConfig | None = None,
    opponent_actor_bundles: Mapping[str, Mapping[str, PolicyMLP]] | None = None,
) -> NilCandidateOutcome:
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    deal_id = f"nil-solver-leaf-{shuffle_seed}"
    auction, nil_count = run_production_single_nil_auction(
        shuffle_seed, bidder, deal_id=deal_id
    )
    if auction is None:
        return NilCandidateOutcome(candidate_index, shuffle_seed, nil_count, None)
    pool_config = opponent_pool_config or OpponentPoolConfig()
    opponent_id = select_opponent_id(
        pool_config, run_seed=run_seed, candidate_index=candidate_index
    )
    opponent_actors: Mapping[str, PolicyMLP] | None = None
    if opponent_id != RULE_OPPONENT_ID:
        if opponent_actor_bundles is None or opponent_id not in opponent_actor_bundles:
            raise RuntimeError(f"Nil opponent actor bundle {opponent_id!r} was not loaded")
        opponent_actors = opponent_actor_bundles[opponent_id]
    room_team0 = play_nil_solver_leaf_room(
        auction,
        actors,
        solver,
        encoder,
        shuffle_seed=shuffle_seed,
        candidate_index=candidate_index,
        candidate_team=0,
        run_seed=run_seed,
        deterministic=deterministic,
        record_transitions=record_transitions,
        opponent_actors=opponent_actors,
    )
    room_team1 = play_nil_solver_leaf_room(
        auction,
        actors,
        solver,
        encoder,
        shuffle_seed=shuffle_seed,
        candidate_index=candidate_index,
        candidate_team=1,
        run_seed=run_seed,
        deterministic=deterministic,
        record_transitions=record_transitions,
        opponent_actors=opponent_actors,
    )
    result = NilDuplicateDealResult(
        deal_id=deal_id,
        candidate_index=candidate_index,
        shuffle_seed=shuffle_seed,
        nil_seat=room_team0.nil_seat,
        opponent_id=opponent_id,
        room_team0=room_team0,
        room_team1=room_team1,
        duplicate_margin_points=(
            room_team0.candidate_margin_points + room_team1.candidate_margin_points
        )
        / 2.0,
        solver_calls=2,
    )
    if record_transitions:
        if len(result.transitions) != 16:
            raise AssertionError("one Nil duplicate deal must contain 16 decisions")
        counts = Counter(item.role for item in result.transitions)
        if counts != Counter({role: 4 for role in NIL_ROLES}):
            raise AssertionError(f"Nil role transitions are unbalanced: {counts}")
    return NilCandidateOutcome(candidate_index, shuffle_seed, 1, result)


@dataclass(slots=True)
class _WorkerRuntime:
    actors: dict[str, PolicyMLP]
    bidder: Any
    solver: ExactDoubleDummyCppFastestSolver
    encoder: NilFirstFourFeatureEncoderV1
    opponent_pool_config: OpponentPoolConfig
    opponent_actor_bundles: dict[str, dict[str, PolicyMLP]]


@dataclass(frozen=True, slots=True)
class _WorkerPayload:
    candidate_indices: tuple[int, ...]
    base_shuffle_seed: int
    run_seed: int
    deterministic: bool
    record_transitions: bool
    actor_state_dicts: dict[str, dict[str, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    outcomes: tuple[NilCandidateOutcome, ...]
    peak_rss_bytes: int


_WORKER_RUNTIME: _WorkerRuntime | None = None


def _build_worker_runtime(
    actor_hidden_dims: tuple[int, ...],
    bid_policy_seed: int | None,
    opponent_pool_config: OpponentPoolConfig,
) -> _WorkerRuntime:
    torch.set_num_threads(1)
    actors = {
        role: PolicyMLP(
            input_dim=NilFirstFourFeatureEncoderV1.TOTAL_DIM,
            hidden_dims=list(actor_hidden_dims),
            output_dim=52,
        ).cpu()
        for role in NIL_ROLES
    }
    bidder = load_deployed_acting_bidder(device="cpu", policy_seed=bid_policy_seed)
    solver = ExactDoubleDummyCppFastestSolver()
    if not solver.native_available:
        raise RuntimeError("原生极速 C++ 求解器不可用，Nil 训练拒绝回退")
    opponent_actor_bundles: dict[str, dict[str, PolicyMLP]] = {}
    if (
        opponent_pool_config.champion_weight > 0.0
        or opponent_pool_config.history_weight > 0.0
    ):
        from residual_bidder.deployment import DEPLOYED_CHECKPOINT_SHA256
        from rl.nil_solver_leaf_ppo import load_nil_role_actor_bundle

        sources: list[tuple[str, str]] = []
        if opponent_pool_config.champion_weight > 0.0:
            assert opponent_pool_config.champion_checkpoint is not None
            sources.append(
                (CHAMPION_OPPONENT_ID, opponent_pool_config.champion_checkpoint)
            )
        if opponent_pool_config.history_weight > 0.0:
            sources.extend(
                (f"history:{index}", checkpoint)
                for index, checkpoint in enumerate(
                    opponent_pool_config.history_checkpoints
                )
            )
        for opponent_id, manifest_path in sources:
            frozen, _, metadata = load_nil_role_actor_bundle(
                Path(manifest_path), device="cpu"
            )
            for role in NIL_ROLES:
                if tuple(metadata[role]["hidden_dims"]) != actor_hidden_dims:
                    raise ValueError(
                        f"Nil opponent {opponent_id!r} architecture differs from learner"
                    )
                if metadata[role].get("residual_checkpoint_sha256") != (
                    DEPLOYED_CHECKPOINT_SHA256
                ):
                    raise ValueError(
                        f"Nil opponent {opponent_id!r} used a different Residual bidder"
                    )
                frozen[role].requires_grad_(False)
            opponent_actor_bundles[opponent_id] = frozen
    return _WorkerRuntime(
        actors=actors,
        bidder=bidder,
        solver=solver,
        encoder=NilFirstFourFeatureEncoderV1(),
        opponent_pool_config=opponent_pool_config,
        opponent_actor_bundles=opponent_actor_bundles,
    )


def _init_worker(
    actor_hidden_dims: tuple[int, ...],
    bid_policy_seed: int | None,
    opponent_pool_config: OpponentPoolConfig,
) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = _build_worker_runtime(
        actor_hidden_dims, bid_policy_seed, opponent_pool_config
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _collect_worker(payload: _WorkerPayload) -> _WorkerResult:
    if _WORKER_RUNTIME is None:
        raise RuntimeError("Nil solver-leaf worker was not initialized")
    runtime = _WORKER_RUNTIME
    if set(payload.actor_state_dicts) != set(NIL_ROLES):
        raise ValueError("worker payload is missing Nil role actor weights")
    for role in NIL_ROLES:
        runtime.actors[role].load_state_dict(payload.actor_state_dicts[role])
        runtime.actors[role].eval()
    outcomes = tuple(
        run_nil_duplicate_candidate(
            candidate_index,
            payload.base_shuffle_seed + candidate_index,
            runtime.actors,
            runtime.bidder,
            runtime.solver,
            runtime.encoder,
            run_seed=payload.run_seed,
            deterministic=payload.deterministic,
            record_transitions=payload.record_transitions,
            opponent_pool_config=runtime.opponent_pool_config,
            opponent_actor_bundles=runtime.opponent_actor_bundles,
        )
        for candidate_index in payload.candidate_indices
    )
    return _WorkerResult(outcomes=outcomes, peak_rss_bytes=_peak_rss_bytes())


def _partition(values: Sequence[int], parts: int) -> list[tuple[int, ...]]:
    buckets: list[list[int]] = [[] for _ in range(min(parts, len(values)))]
    for index, value in enumerate(values):
        buckets[index % len(buckets)].append(value)
    return [tuple(bucket) for bucket in buckets if bucket]


class NilProductionDuplicateCollector:
    """Persistent bidder/solver workers for the four Nil role policies."""

    def __init__(
        self,
        *,
        workers: int,
        actor_hidden_dims: Sequence[int] = (1024, 512, 512),
        bid_policy_seed: int | None = None,
        opponent_pool_config: OpponentPoolConfig | None = None,
        oversample_factor: float = 7.0,
        minimum_scan: int = 32,
    ) -> None:
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if not actor_hidden_dims or any(
            type(value) is not int or value <= 0 for value in actor_hidden_dims
        ):
            raise ValueError("actor_hidden_dims must contain positive integers")
        if not math.isfinite(oversample_factor) or oversample_factor < 1.0:
            raise ValueError("oversample_factor must be finite and at least one")
        if type(minimum_scan) is not int or minimum_scan <= 0:
            raise ValueError("minimum_scan must be positive")
        self.workers = workers
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.bid_policy_seed = bid_policy_seed
        self.opponent_pool_config = opponent_pool_config or OpponentPoolConfig()
        self.oversample_factor = float(oversample_factor)
        self.minimum_scan = minimum_scan
        self._pool: Any = None
        self._local_runtime: _WorkerRuntime | None = None

    def __enter__(self) -> "NilProductionDuplicateCollector":
        if self.workers == 1:
            self._local_runtime = _build_worker_runtime(
                self.actor_hidden_dims,
                self.bid_policy_seed,
                self.opponent_pool_config,
            )
        else:
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                processes=self.workers,
                initializer=_init_worker,
                initargs=(
                    self.actor_hidden_dims,
                    self.bid_policy_seed,
                    self.opponent_pool_config,
                ),
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, traceback
        if self._pool is not None:
            if exc is None:
                self._pool.close()
            else:
                self._pool.terminate()
            self._pool.join()
            self._pool = None
        self._local_runtime = None

    def _run_payloads(self, payloads: Sequence[_WorkerPayload]) -> list[_WorkerResult]:
        if self.workers == 1:
            if self._local_runtime is None:
                raise RuntimeError("collector must be used as a context manager")
            global _WORKER_RUNTIME
            previous = _WORKER_RUNTIME
            try:
                _WORKER_RUNTIME = self._local_runtime
                return [_collect_worker(payloads[0])]
            finally:
                _WORKER_RUNTIME = previous
        if self._pool is None:
            raise RuntimeError("collector must be used as a context manager")
        return list(self._pool.map(_collect_worker, payloads))

    def collect(
        self,
        actors: Mapping[str, PolicyMLP],
        *,
        start_candidate_index: int,
        target_deals: int,
        base_shuffle_seed: int,
        run_seed: int,
        deterministic: bool,
        record_transitions: bool,
    ) -> NilCollectionBatch:
        if set(actors) != set(NIL_ROLES):
            raise ValueError("actors must contain exactly the four Nil roles")
        if type(start_candidate_index) is not int or start_candidate_index < 0:
            raise ValueError("start_candidate_index must be nonnegative")
        if type(target_deals) is not int or target_deals <= 0:
            raise ValueError("target_deals must be positive")
        if type(base_shuffle_seed) is not int or base_shuffle_seed < 0:
            raise ValueError("base_shuffle_seed must be nonnegative")
        if type(run_seed) is not int or run_seed < 0:
            raise ValueError("run_seed must be nonnegative")
        actor_state_dicts = {
            role: {
                name: tensor.detach().cpu().clone()
                for name, tensor in actors[role].state_dict().items()
            }
            for role in NIL_ROLES
        }
        started = time.perf_counter()
        cursor = start_candidate_index
        accepted: list[NilDuplicateDealResult] = []
        scanned = 0
        nil_counts: Counter[int] = Counter()
        peak_rss = 0
        aggregate_peak_rss = 0
        max_scanned = max(10_000, target_deals * 100)

        while len(accepted) < target_deals:
            needed = target_deals - len(accepted)
            scan_count = max(
                self.minimum_scan,
                int(math.ceil(needed * self.oversample_factor)),
            )
            indices = tuple(range(cursor, cursor + scan_count))
            payloads = [
                _WorkerPayload(
                    candidate_indices=partition,
                    base_shuffle_seed=base_shuffle_seed,
                    run_seed=run_seed,
                    deterministic=deterministic,
                    record_transitions=record_transitions,
                    actor_state_dicts=actor_state_dicts,
                )
                for partition in _partition(indices, self.workers)
            ]
            worker_results = self._run_payloads(payloads)
            peak_rss = max(peak_rss, *(item.peak_rss_bytes for item in worker_results))
            aggregate_peak_rss = max(
                aggregate_peak_rss,
                sum(item.peak_rss_bytes for item in worker_results),
            )
            outcomes = sorted(
                (item for result in worker_results for item in result.outcomes),
                key=lambda item: item.candidate_index,
            )
            selected_last: int | None = None
            for outcome in outcomes:
                scanned += 1
                nil_counts[outcome.nil_count] += 1
                if outcome.result is not None:
                    accepted.append(outcome.result)
                if len(accepted) == target_deals:
                    selected_last = outcome.candidate_index
                    break
            if selected_last is not None:
                cursor = selected_last + 1
                break
            cursor += scan_count
            if scanned > max_scanned:
                raise RuntimeError("too many deals lacked exactly one Nil before collection")

        return NilCollectionBatch(
            deals=tuple(accepted),
            start_candidate_index=start_candidate_index,
            next_candidate_index=cursor,
            scanned_candidates=scanned,
            nil_count_histogram=dict(sorted(nil_counts.items())),
            elapsed_seconds=time.perf_counter() - started,
            worker_peak_rss_bytes=peak_rss,
            aggregate_worker_peak_rss_bytes=aggregate_peak_rss,
        )


__all__ = [
    "NIL_LOWER",
    "NIL_PARTNER",
    "NIL_ROLES",
    "NIL_SELF",
    "NIL_UPPER",
    "NilCandidateOutcome",
    "NilCollectionBatch",
    "NilDuplicateDealResult",
    "NilLeafTransition",
    "NilProductionDuplicateCollector",
    "NilRoomResult",
    "play_nil_solver_leaf_room",
    "role_for_seat",
    "run_nil_duplicate_candidate",
    "run_production_single_nil_auction",
]
