"""Paired first-four rollout environment with one exact leaf solve per room."""

from __future__ import annotations

import copy
import hashlib
import math
import multiprocessing as mp
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.deployment import load_deployed_acting_bidder
from rl.first4_observation import (
    FirstFourFeatureEncoderV2,
    build_first_four_observation,
)
from rl.policy_network import PolicyMLP
from strategy.rule_based_first4_player import RuleBasedFirst4Player
from strategy.spades_match_runner import build_random_state
from trick_taking.card import Suit
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


FIRST_TRICKS = 4
TARGET_DIVISOR = 100.0
ROOM_CANDIDATE_TEAM_0 = 0
ROOM_CANDIDATE_TEAM_1 = 1
RULE_OPPONENT_ID = "rule"
CHAMPION_OPPONENT_ID = "champion"


@dataclass(frozen=True, slots=True)
class OpponentPoolConfig:
    """Deterministic per-deal mixture of blind first-four opponents."""

    rule_weight: float = 1.0
    champion_weight: float = 0.0
    history_weight: float = 0.0
    champion_checkpoint: str | None = None
    history_checkpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        weights = (self.rule_weight, self.champion_weight, self.history_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("opponent weights must be finite and nonnegative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one opponent weight must be positive")
        if self.champion_weight > 0.0 and not self.champion_checkpoint:
            raise ValueError("positive champion weight requires a champion checkpoint")
        if self.history_weight > 0.0 and not self.history_checkpoints:
            raise ValueError("positive history weight requires history checkpoints")
        checkpoints = tuple(
            path
            for path in (self.champion_checkpoint, *self.history_checkpoints)
            if path is not None
        )
        if any(not isinstance(path, str) or not path for path in checkpoints):
            raise ValueError("opponent checkpoint paths must be nonempty strings")
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("opponent checkpoint paths must be unique")


def select_opponent_id(
    config: OpponentPoolConfig,
    *,
    run_seed: int,
    candidate_index: int,
) -> str:
    """Select an opponent reproducibly, independently of worker scheduling."""

    if type(run_seed) is not int or run_seed < 0:
        raise ValueError("run_seed must be a nonnegative integer")
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    digest = hashlib.blake2b(
        f"opponent:{run_seed}:{candidate_index}".encode("ascii"), digest_size=16
    ).digest()
    draw = int.from_bytes(digest[:8], "little") / float(1 << 64)
    total = config.rule_weight + config.champion_weight + config.history_weight
    scaled = draw * total
    if scaled < config.rule_weight:
        return RULE_OPPONENT_ID
    if scaled < config.rule_weight + config.champion_weight:
        return CHAMPION_OPPONENT_ID
    history_index = int.from_bytes(digest[8:], "little") % len(
        config.history_checkpoints
    )
    return f"history:{history_index}"


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
class LeafTransition:
    feature: np.ndarray
    legal_mask: np.ndarray
    action: int
    old_log_prob: float
    old_entropy: float
    reward: float
    candidate_index: int
    candidate_team: int
    seat: int
    play_index: int


@dataclass(frozen=True, slots=True)
class RoomResult:
    candidate_team: int
    team0_margin_points: float
    candidate_margin_points: float
    reward: float
    transitions: tuple[LeafTransition, ...]
    solver_seconds: float


@dataclass(frozen=True, slots=True)
class DuplicateDealResult:
    deal_id: str
    candidate_index: int
    shuffle_seed: int
    opponent_id: str
    room_team0: RoomResult
    room_team1: RoomResult
    duplicate_margin_points: float
    solver_calls: int

    @property
    def transitions(self) -> tuple[LeafTransition, ...]:
        return self.room_team0.transitions + self.room_team1.transitions


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    candidate_index: int
    shuffle_seed: int
    nil_filtered: bool
    result: DuplicateDealResult | None

    def __post_init__(self) -> None:
        if self.nil_filtered == (self.result is not None):
            raise ValueError("a candidate must be either Nil-filtered or accepted")


@dataclass(frozen=True, slots=True)
class CollectionBatch:
    deals: tuple[DuplicateDealResult, ...]
    start_candidate_index: int
    next_candidate_index: int
    scanned_candidates: int
    nil_filtered_candidates: int
    elapsed_seconds: float
    worker_peak_rss_bytes: int
    aggregate_worker_peak_rss_bytes: int

    @property
    def transitions(self) -> tuple[LeafTransition, ...]:
        return tuple(transition for deal in self.deals for transition in deal.transitions)

    @property
    def solver_calls(self) -> int:
        return sum(deal.solver_calls for deal in self.deals)


def legal_mask_from_ids(card_ids: Sequence[int]) -> np.ndarray:
    mask = np.zeros(52, dtype=np.bool_)
    for card_id in card_ids:
        if not 0 <= int(card_id) < 52:
            raise ValueError("legal card ids must be in [0, 51]")
        mask[int(card_id)] = True
    if not bool(mask.any()):
        raise ValueError("at least one action must be legal")
    return mask


def mask_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mask 52 policy logits without remapping card-id slots."""

    if not isinstance(logits, torch.Tensor) or logits.shape[-1:] != (52,):
        raise ValueError("policy logits must end in dimension 52")
    if not isinstance(legal_mask, torch.Tensor) or legal_mask.shape != logits.shape:
        raise ValueError("legal mask must have the same shape as logits")
    if legal_mask.dtype is not torch.bool:
        raise TypeError("legal mask must be boolean")
    if not bool(legal_mask.any(dim=-1).all().item()):
        raise ValueError("every policy row must contain at least one legal action")
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def masked_action_probabilities(
    actor: PolicyMLP,
    features: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    logits = actor(features)
    return torch.softmax(mask_policy_logits(logits, legal_mask), dim=-1)


def derive_action_seed(
    run_seed: int,
    candidate_index: int,
    candidate_team: int,
    seat: int,
    play_index: int,
) -> int:
    """Derive sampling randomness independently of worker scheduling."""

    values = (run_seed, candidate_index, candidate_team, seat, play_index)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("action-seed fields must be nonnegative integers")
    payload = ":".join(str(value) for value in values).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & (
        (1 << 63) - 1
    )


def select_policy_action(
    actor: PolicyMLP,
    feature: np.ndarray,
    legal_mask: np.ndarray,
    *,
    deterministic: bool,
    sample_seed: int,
) -> tuple[int, float, float]:
    if feature.shape != (FirstFourFeatureEncoderV2.TOTAL_DIM,):
        raise ValueError("feature must have shape (536,)")
    if legal_mask.shape != (52,) or legal_mask.dtype != np.bool_:
        raise ValueError("legal_mask must be bool with shape (52,)")
    feature_tensor = torch.from_numpy(feature).to(dtype=torch.float32)
    mask_tensor = torch.from_numpy(legal_mask)
    with torch.inference_mode():
        logits = actor(feature_tensor)
        masked_logits = mask_policy_logits(logits, mask_tensor)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        if deterministic:
            action_tensor = torch.argmax(masked_logits)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
            action_tensor = torch.multinomial(
                distribution.probs, 1, generator=generator
            ).squeeze(0)
        log_prob = float(distribution.log_prob(action_tensor).item())
        entropy = float(distribution.entropy().item())
        action = int(action_tensor.item())
    if not legal_mask[action] or not math.isfinite(log_prob) or not math.isfinite(entropy):
        raise RuntimeError("policy produced an invalid masked action")
    return action, log_prob, entropy


def run_production_auction(
    shuffle_seed: int,
    bidder: ActingBidder,
    *,
    deal_id: str,
) -> GameState | None:
    """Run one shared production auction; return ``None`` for a Nil deal."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if not isinstance(deal_id, str) or not deal_id:
        raise ValueError("deal_id must be nonempty")
    if not callable(getattr(bidder, "choose", None)):
        raise TypeError("bidder must provide choose")

    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state = build_random_state(shuffle_seed)
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
        if action is BidAction.NIL:
            return None
        state.bids.append(Bid(player_id=seat, value=value, is_pass=False))
        state.max_bid[seat] = value
        state.current_bidder = rules.next_bid_turn(state)

    if not rules.end_bidding(state) or any(value is None for value in state.max_bid):
        raise AssertionError("production auction did not produce four actual bids")
    state.teams = rules.set_team(state)
    state.points = rules.initial_points(state)
    return state


def assert_solver_leaf_boundary(state: GameState) -> None:
    if state.phase is not Phase.PLAYING:
        raise AssertionError("solver leaf must remain in Phase.PLAYING")
    if state.tricks_played != FIRST_TRICKS:
        raise AssertionError("solver leaf must complete exactly four tricks")
    if state.table_cards:
        raise AssertionError("solver leaf table must be empty")
    if tuple(len(hand) for hand in state.hands) != (9, 9, 9, 9):
        raise AssertionError("solver leaf must leave nine cards in every hand")
    if sum(len(hand) for hand in state.hands) != 36:
        raise AssertionError("solver leaf must leave 36 cards in total")


def _rule_players(
    state: GameState,
    *,
    candidate_team: int,
    shuffle_seed: int,
) -> dict[int, RuleBasedFirst4Player]:
    players: dict[int, RuleBasedFirst4Player] = {}
    for seat in range(state.num_players):
        if state.teams[seat] == candidate_team:
            continue
        player = RuleBasedFirst4Player(
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


def play_solver_leaf_room(
    auction_state: GameState,
    actor: PolicyMLP,
    solver: TerminalSolver,
    encoder: FirstFourFeatureEncoderV2,
    *,
    shuffle_seed: int,
    candidate_index: int,
    candidate_team: int,
    run_seed: int,
    deterministic: bool,
    record_transitions: bool,
    opponent_actor: PolicyMLP | None = None,
) -> RoomResult:
    """Play four blind tricks and solve the revealed leaf exactly once."""

    if candidate_team not in (0, 1):
        raise ValueError("candidate_team must be zero or one")
    state = copy.deepcopy(auction_state)
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    rules_by_seat = (
        _rule_players(state, candidate_team=candidate_team, shuffle_seed=shuffle_seed)
        if opponent_actor is None
        else {}
    )
    actor.eval()
    if opponent_actor is not None:
        opponent_actor.eval()
    pending: list[tuple[np.ndarray, np.ndarray, int, float, float, int, int]] = []
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
                observation = build_first_four_observation(state, seat, legal)
                feature = encoder.encode(observation)
                legal_mask = legal_mask_from_ids(observation.legal_card_ids)
                feature_mask = feature[
                    encoder.LEGAL_START : encoder.LEGAL_START + 52
                ].astype(np.bool_)
                if not np.array_equal(legal_mask, feature_mask):
                    raise AssertionError("input legal mask disagrees with action mask")
                action, log_prob, entropy = select_policy_action(
                    actor,
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
                    raise RuntimeError("masked policy selected an illegal card")
                if record_transitions:
                    pending.append(
                        (
                            feature.copy(),
                            legal_mask.copy(),
                            action,
                            log_prob,
                            entropy,
                            seat,
                            play_index,
                        )
                    )
            else:
                if opponent_actor is None:
                    player = rules_by_seat[seat]
                    card = player.play_card(legal, state.get_player_view(seat))
                else:
                    observation = build_first_four_observation(state, seat, legal)
                    feature = encoder.encode(observation)
                    legal_mask = legal_mask_from_ids(observation.legal_card_ids)
                    action, _, _ = select_policy_action(
                        opponent_actor,
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
                        raise RuntimeError("masked opponent policy selected an illegal card")
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
        LeafTransition(
            feature=feature,
            legal_mask=legal_mask,
            action=action,
            old_log_prob=old_log_prob,
            old_entropy=old_entropy,
            reward=reward,
            candidate_index=candidate_index,
            candidate_team=candidate_team,
            seat=seat,
            play_index=transition_play_index,
        )
        for (
            feature,
            legal_mask,
            action,
            old_log_prob,
            old_entropy,
            seat,
            transition_play_index,
        ) in pending
    )
    if record_transitions and len(transitions) != 8:
        raise AssertionError("each room must contain eight candidate decisions")
    return RoomResult(
        candidate_team=candidate_team,
        team0_margin_points=team0_margin,
        candidate_margin_points=candidate_margin,
        reward=reward,
        transitions=transitions,
        solver_seconds=solver_seconds,
    )


def run_duplicate_candidate(
    candidate_index: int,
    shuffle_seed: int,
    actor: PolicyMLP,
    bidder: ActingBidder,
    solver: TerminalSolver,
    encoder: FirstFourFeatureEncoderV2,
    *,
    run_seed: int,
    deterministic: bool,
    record_transitions: bool,
    opponent_pool_config: OpponentPoolConfig | None = None,
    opponent_actors: Mapping[str, PolicyMLP] | None = None,
) -> CandidateOutcome:
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    deal_id = f"solver-leaf-{shuffle_seed}"
    auction = run_production_auction(shuffle_seed, bidder, deal_id=deal_id)
    if auction is None:
        return CandidateOutcome(candidate_index, shuffle_seed, True, None)
    pool_config = opponent_pool_config or OpponentPoolConfig()
    opponent_id = select_opponent_id(
        pool_config, run_seed=run_seed, candidate_index=candidate_index
    )
    opponent_actor: PolicyMLP | None = None
    if opponent_id != RULE_OPPONENT_ID:
        if opponent_actors is None or opponent_id not in opponent_actors:
            raise RuntimeError(f"opponent actor {opponent_id!r} was not loaded")
        opponent_actor = opponent_actors[opponent_id]
    room_team0 = play_solver_leaf_room(
        auction,
        actor,
        solver,
        encoder,
        shuffle_seed=shuffle_seed,
        candidate_index=candidate_index,
        candidate_team=ROOM_CANDIDATE_TEAM_0,
        run_seed=run_seed,
        deterministic=deterministic,
        record_transitions=record_transitions,
        opponent_actor=opponent_actor,
    )
    room_team1 = play_solver_leaf_room(
        auction,
        actor,
        solver,
        encoder,
        shuffle_seed=shuffle_seed,
        candidate_index=candidate_index,
        candidate_team=ROOM_CANDIDATE_TEAM_1,
        run_seed=run_seed,
        deterministic=deterministic,
        record_transitions=record_transitions,
        opponent_actor=opponent_actor,
    )
    duplicate_margin = (
        room_team0.candidate_margin_points + room_team1.candidate_margin_points
    ) / 2.0
    result = DuplicateDealResult(
        deal_id=deal_id,
        candidate_index=candidate_index,
        shuffle_seed=shuffle_seed,
        opponent_id=opponent_id,
        room_team0=room_team0,
        room_team1=room_team1,
        duplicate_margin_points=duplicate_margin,
        solver_calls=2,
    )
    if record_transitions and len(result.transitions) != 16:
        raise AssertionError("one duplicate deal must contain sixteen candidate decisions")
    return CandidateOutcome(candidate_index, shuffle_seed, False, result)


@dataclass(slots=True)
class _WorkerRuntime:
    actor: PolicyMLP
    bidder: Any
    solver: ExactDoubleDummyCppFastestSolver
    encoder: FirstFourFeatureEncoderV2
    opponent_pool_config: OpponentPoolConfig
    opponent_actors: dict[str, PolicyMLP]


@dataclass(frozen=True, slots=True)
class _WorkerPayload:
    candidate_indices: tuple[int, ...]
    base_shuffle_seed: int
    run_seed: int
    deterministic: bool
    record_transitions: bool
    actor_state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    outcomes: tuple[CandidateOutcome, ...]
    peak_rss_bytes: int


_WORKER_RUNTIME: _WorkerRuntime | None = None


def _build_worker_runtime(
    actor_hidden_dims: tuple[int, ...],
    bid_policy_seed: int | None,
    opponent_pool_config: OpponentPoolConfig,
) -> _WorkerRuntime:
    torch.set_num_threads(1)
    actor = PolicyMLP(
        input_dim=FirstFourFeatureEncoderV2.TOTAL_DIM,
        hidden_dims=list(actor_hidden_dims),
        output_dim=52,
    ).cpu()
    bidder = load_deployed_acting_bidder(device="cpu", policy_seed=bid_policy_seed)
    solver = ExactDoubleDummyCppFastestSolver()
    if not solver.native_available:
        raise RuntimeError("原生极速 C++ 求解器不可用，训练拒绝回退到 Python solver")
    opponent_actors: dict[str, PolicyMLP] = {}
    if (
        opponent_pool_config.champion_weight > 0.0
        or opponent_pool_config.history_weight > 0.0
    ):
        from residual_bidder.deployment import DEPLOYED_CHECKPOINT_SHA256
        from rl.solver_leaf_ppo import load_exported_actor

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
        for opponent_id, checkpoint in sources:
            frozen_actor, metadata = load_exported_actor(Path(checkpoint), device="cpu")
            if tuple(metadata["hidden_dims"]) != actor_hidden_dims:
                raise ValueError(
                    f"opponent {opponent_id!r} architecture differs from learner"
                )
            if metadata.get("residual_checkpoint_sha256") != DEPLOYED_CHECKPOINT_SHA256:
                raise ValueError(
                    f"opponent {opponent_id!r} used a different Residual bidder"
                )
            frozen_actor.requires_grad_(False)
            opponent_actors[opponent_id] = frozen_actor
    return _WorkerRuntime(
        actor,
        bidder,
        solver,
        FirstFourFeatureEncoderV2(),
        opponent_pool_config,
        opponent_actors,
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
        raise RuntimeError("solver-leaf worker was not initialized")
    runtime = _WORKER_RUNTIME
    runtime.actor.load_state_dict(payload.actor_state_dict)
    runtime.actor.eval()
    outcomes = tuple(
        run_duplicate_candidate(
            candidate_index,
            payload.base_shuffle_seed + candidate_index,
            runtime.actor,
            runtime.bidder,
            runtime.solver,
            runtime.encoder,
            run_seed=payload.run_seed,
            deterministic=payload.deterministic,
            record_transitions=payload.record_transitions,
            opponent_pool_config=runtime.opponent_pool_config,
            opponent_actors=runtime.opponent_actors,
        )
        for candidate_index in payload.candidate_indices
    )
    return _WorkerResult(outcomes, _peak_rss_bytes())


def _partition(values: Sequence[int], parts: int) -> list[tuple[int, ...]]:
    buckets: list[list[int]] = [[] for _ in range(min(parts, len(values)))]
    for index, value in enumerate(values):
        buckets[index % len(buckets)].append(value)
    return [tuple(bucket) for bucket in buckets if bucket]


class ProductionDuplicateCollector:
    """Persistent production bidder/solver workers for PPO and evaluation."""

    def __init__(
        self,
        *,
        workers: int,
        actor_hidden_dims: Sequence[int] = (1024, 512, 512),
        bid_policy_seed: int | None = None,
        opponent_pool_config: OpponentPoolConfig | None = None,
        oversample_factor: float = 1.25,
        minimum_scan: int = 32,
    ) -> None:
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if not actor_hidden_dims or any(type(value) is not int or value <= 0 for value in actor_hidden_dims):
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

    def __enter__(self) -> ProductionDuplicateCollector:
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
        actor: PolicyMLP,
        *,
        start_candidate_index: int,
        target_deals: int,
        base_shuffle_seed: int,
        run_seed: int,
        deterministic: bool,
        record_transitions: bool,
    ) -> CollectionBatch:
        if type(start_candidate_index) is not int or start_candidate_index < 0:
            raise ValueError("start_candidate_index must be nonnegative")
        if type(target_deals) is not int or target_deals <= 0:
            raise ValueError("target_deals must be positive")
        if type(base_shuffle_seed) is not int or base_shuffle_seed < 0:
            raise ValueError("base_shuffle_seed must be nonnegative")
        if type(run_seed) is not int or run_seed < 0:
            raise ValueError("run_seed must be nonnegative")
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in actor.state_dict().items()
        }
        started = time.perf_counter()
        cursor = start_candidate_index
        accepted: list[DuplicateDealResult] = []
        scanned = 0
        nil_filtered = 0
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
                    actor_state_dict=state_dict,
                )
                for partition in _partition(indices, self.workers)
            ]
            worker_results = self._run_payloads(payloads)
            peak_rss = max(peak_rss, *(result.peak_rss_bytes for result in worker_results))
            aggregate_peak_rss = max(
                aggregate_peak_rss,
                sum(result.peak_rss_bytes for result in worker_results),
            )
            outcomes = sorted(
                (outcome for result in worker_results for outcome in result.outcomes),
                key=lambda item: item.candidate_index,
            )
            selected_last: int | None = None
            for outcome in outcomes:
                scanned += 1
                if outcome.nil_filtered:
                    nil_filtered += 1
                else:
                    assert outcome.result is not None
                    accepted.append(outcome.result)
                if len(accepted) == target_deals:
                    selected_last = outcome.candidate_index
                    break
            if selected_last is not None:
                cursor = selected_last + 1
                break
            cursor += scan_count
            if scanned > max_scanned:
                raise RuntimeError("too many candidate deals were filtered before collection")

        return CollectionBatch(
            deals=tuple(accepted),
            start_candidate_index=start_candidate_index,
            next_candidate_index=cursor,
            scanned_candidates=scanned,
            nil_filtered_candidates=nil_filtered,
            elapsed_seconds=time.perf_counter() - started,
            worker_peak_rss_bytes=peak_rss,
            aggregate_worker_peak_rss_bytes=aggregate_peak_rss,
        )


__all__ = [
    "FIRST_TRICKS",
    "TARGET_DIVISOR",
    "CandidateOutcome",
    "CollectionBatch",
    "DuplicateDealResult",
    "LeafTransition",
    "OpponentPoolConfig",
    "ProductionDuplicateCollector",
    "RoomResult",
    "assert_solver_leaf_boundary",
    "derive_action_seed",
    "legal_mask_from_ids",
    "mask_policy_logits",
    "masked_action_probabilities",
    "play_solver_leaf_room",
    "run_duplicate_candidate",
    "run_production_auction",
    "select_opponent_id",
    "select_policy_action",
]
