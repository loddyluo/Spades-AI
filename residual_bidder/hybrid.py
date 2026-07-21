"""Minimal four-trick-plus-DDS counterfactual data generation."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch

from residual_bidder.actions import BidAction, neighborhood, to_local_bid
from residual_bidder.model import INPUT_DIM, build_residual_input
from residual_bidder.nsfp import NSFPObservation
from strategy.rule_based_first4_nil_player import RuleBasedFirst4NilPlayer
from strategy.rule_based_first4_player import RuleBasedFirst4Player
from trick_taking.deck import Deck
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules


TARGET_DIVISOR = 100.0
FIRST_TRICKS = 4
ROOM_CANDIDATE_TEAM_0 = 0
ROOM_CANDIDATE_TEAM_1 = 1


class NSFPObserver(Protocol):
    def observe(self, state: GameState) -> NSFPObservation: ...


class TerminalSolver(Protocol):
    def solve(self, state: GameState) -> float: ...


@dataclass(frozen=True)
class HybridTrainingRow:
    """One baseline bidding observation and its two local advantages."""

    deal_id: str
    shuffle_seed: int
    room_id: int
    physical_seat: int
    bid_index: int
    center: BidAction
    features: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    baseline_margin: float


@dataclass(frozen=True)
class HybridDeal:
    """All four bidding rows generated from one duplicate deal."""

    deal_id: str
    shuffle_seed: int
    rows: tuple[HybridTrainingRow, ...]
    solver_calls: int


_ARRAY_NAMES = (
    "features",
    "targets",
    "masks",
    "centers",
    "baseline_margins",
    "shuffle_seeds",
    "room_ids",
    "physical_seats",
    "bid_indices",
    "deal_ids",
)


@dataclass(frozen=True)
class HybridArrays:
    """Pickle-free arrays written by the minimal data command."""

    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    centers: np.ndarray
    baseline_margins: np.ndarray
    shuffle_seeds: np.ndarray
    room_ids: np.ndarray
    physical_seats: np.ndarray
    bid_indices: np.ndarray
    deal_ids: np.ndarray

    def array_names(self) -> tuple[str, ...]:
        return _ARRAY_NAMES


@dataclass(frozen=True)
class _Auction:
    state: GameState
    observations: tuple[NSFPObservation, ...]
    physical_seats: tuple[int, ...]


def _initial_state(shuffle_seed: int, rules: SpadesRules) -> GameState:
    deck = Deck(rules.deck_config, seed=shuffle_seed)
    hands = [deck.deal(rules.cards_per_hand) for _ in range(rules.num_players)]
    state = GameState()
    state.init_for_deal(rules.num_players, hands, [], deck.all_cards)
    state.dealer_seat = random.Random(shuffle_seed).randrange(rules.num_players)
    opener = (state.dealer_seat + 1) % rules.num_players
    state.current_bidder = opener
    state.turn = opener
    state.trick_leader = opener
    state.phase = Phase.BIDDING
    return state


def _run_auction(
    initial: GameState,
    nsfp: NSFPObserver,
    rules: SpadesRules,
    *,
    forced_bid_index: int | None = None,
    forced_action: BidAction | None = None,
) -> _Auction:
    if (forced_bid_index is None) != (forced_action is None):
        raise ValueError("forced bid index and action must be supplied together")
    if forced_bid_index is not None and not 0 <= forced_bid_index < rules.num_players:
        raise ValueError("forced bid index must be between zero and three")

    state = copy.deepcopy(initial)
    observations: list[NSFPObservation] = []
    physical_seats: list[int] = []
    for bid_index in range(rules.num_players):
        bidder = state.current_bidder
        observation = nsfp.observe(state)
        if not isinstance(observation, NSFPObservation):
            raise TypeError("NSFP observer must return NSFPObservation")
        observations.append(observation)
        physical_seats.append(bidder)
        action = (
            forced_action
            if forced_bid_index is not None and bid_index == forced_bid_index
            else observation.center
        )
        assert action is not None
        value = to_local_bid(action)
        legal = rules.legal_bids(state, bidder)
        if value not in legal:
            raise ValueError(f"forced or NSFP bid {value!r} is illegal for seat {bidder}")
        state.bids.append(Bid(player_id=bidder, value=value, is_pass=False))
        state.max_bid[bidder] = value
        state.current_bidder = rules.next_bid_turn(state)

    if not rules.end_bidding(state) or any(value is None for value in state.max_bid):
        raise AssertionError("baseline/branch auction did not produce four actual bids")
    state.teams = rules.set_team(state)
    state.points = rules.initial_points(state)
    return _Auction(state, tuple(observations), tuple(physical_seats))


def _players_for_branch(state: GameState, shuffle_seed: int) -> list[Any]:
    has_nil = any(value == "nil" for value in state.max_bid)
    player_type = RuleBasedFirst4NilPlayer if has_nil else RuleBasedFirst4Player
    players = [
        player_type(bid_seed=(shuffle_seed & 0x7FFFFFFF) + seat)
        for seat in range(state.num_players)
    ]
    for seat, player in enumerate(players):
        player.start_game(seat, list(state.hands[seat]), state.num_players)
    for record in state.bids:
        for player in players:
            player.bid_placed(record.player_id, record.value)
    for player in players:
        # The Nil rule player indexes bids by physical seat, so this must be
        # seat-ordered rather than chronological dealer order.
        player.set_teams(list(state.teams), list(state.max_bid))
    return players


def _play_first_four(state: GameState, rules: SpadesRules, shuffle_seed: int) -> None:
    players = _players_for_branch(state, shuffle_seed)
    state.phase = Phase.PLAYING
    trump_suits = rules.trump_mask(state)
    if trump_suits is None or len(trump_suits) != 1:
        raise AssertionError("Spades branch must have exactly one trump suit")
    state.trump_suit = next(iter(trump_suits))

    for _ in range(FIRST_TRICKS):
        if state.table_cards:
            raise AssertionError("table must be empty at trick start")
        for _ in range(rules.num_players):
            current = state.turn
            legal = rules.playable(state, state.hands[current], current)
            if not legal:
                raise AssertionError(f"seat {current} has no legal card")
            card = players[current].play_card(legal, state.get_player_view(current))
            if card not in legal:
                raise ValueError(f"seat {current} returned an illegal card")
            state.play_card_to_table(current, card)
            if card.suit == state.trump_suit:
                state.trump_broken = True
                state.spades_broken = True
            for player in players:
                player.card_played(current, card)
            state.turn = (current + 1) % rules.num_players
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.turn = winner
        state.trick_leader = winner

    if state.tricks_played != FIRST_TRICKS:
        raise AssertionError("hybrid branch did not complete exactly four tricks")
    if state.table_cards:
        raise AssertionError("hybrid branch ended with a nonempty table")
    if tuple(len(hand) for hand in state.hands) != (9, 9, 9, 9):
        raise AssertionError("hybrid branch must leave nine cards in every hand")
    if sum(len(hand) for hand in state.hands) != 36:
        raise AssertionError("hybrid branch must leave 36 cards")


def _evaluate_branch(
    auction: _Auction,
    solver: TerminalSolver,
    rules: SpadesRules,
    shuffle_seed: int,
) -> float:
    _play_first_four(auction.state, rules, shuffle_seed)
    value = float(solver.solve(auction.state))
    if not math.isfinite(value):
        raise ValueError("terminal solver returned a non-finite value")
    return value


def generate_hybrid_deal(
    shuffle_seed: int,
    nsfp: NSFPObserver,
    solver: TerminalSolver,
    *,
    deal_id: str | None = None,
) -> HybridDeal:
    """Generate four local-advantage rows from one deterministic deal."""

    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    if not callable(getattr(nsfp, "observe", None)):
        raise TypeError("nsfp must provide observe(state)")
    if not callable(getattr(solver, "solve", None)):
        raise TypeError("solver must provide solve(state)")
    resolved_deal_id = deal_id or f"deal-{shuffle_seed}"
    if not isinstance(resolved_deal_id, str) or not resolved_deal_id:
        raise ValueError("deal_id must be a nonempty string")

    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    initial = _initial_state(shuffle_seed, rules)
    baseline = _run_auction(initial, nsfp, rules)
    baseline_team0_margin = _evaluate_branch(baseline, solver, rules, shuffle_seed)
    solver_calls = 1

    rows: list[HybridTrainingRow] = []
    for bid_index, (observation, physical_seat) in enumerate(
        zip(baseline.observations, baseline.physical_seats, strict=True)
    ):
        residual_input = build_residual_input(observation)
        local = neighborhood(observation.center)
        team = physical_seat % 2
        perspective_sign = 1.0 if team == 0 else -1.0
        baseline_margin = perspective_sign * baseline_team0_margin
        targets = torch.zeros(2, dtype=torch.float32)

        for slot, alternative in enumerate((local.lower, local.upper)):
            if alternative is None:
                continue
            branch = _run_auction(
                initial,
                nsfp,
                rules,
                forced_bid_index=bid_index,
                forced_action=alternative,
            )
            branch_team0_margin = _evaluate_branch(branch, solver, rules, shuffle_seed)
            solver_calls += 1
            branch_margin = perspective_sign * branch_team0_margin
            targets[slot] = (branch_margin - baseline_margin) / TARGET_DIVISOR

        rows.append(
            HybridTrainingRow(
                deal_id=resolved_deal_id,
                shuffle_seed=shuffle_seed,
                room_id=(ROOM_CANDIDATE_TEAM_0 if team == 0 else ROOM_CANDIDATE_TEAM_1),
                physical_seat=physical_seat,
                bid_index=bid_index,
                center=observation.center,
                features=residual_input.values.detach().cpu().to(dtype=torch.float32).clone(),
                targets=targets,
                mask=residual_input.alternative_mask.detach()
                .cpu()
                .to(dtype=torch.float32)
                .clone(),
                baseline_margin=float(baseline_margin),
            )
        )

    if len(rows) != 4 or {row.physical_seat for row in rows} != {0, 1, 2, 3}:
        raise AssertionError("one hybrid deal must produce one row for every physical seat")
    return HybridDeal(resolved_deal_id, shuffle_seed, tuple(rows), solver_calls)


def stack_hybrid_deals(deals: Sequence[HybridDeal]) -> HybridArrays:
    """Stack complete deals without splitting their four rows."""

    if not isinstance(deals, Sequence) or not deals:
        raise ValueError("deals must be a nonempty sequence")
    rows = [row for deal in deals for row in deal.rows]
    if any(len(deal.rows) != 4 for deal in deals):
        raise ValueError("every deal must contain exactly four rows")
    features = torch.stack([row.features for row in rows]).numpy().astype(np.float32, copy=False)
    targets = torch.stack([row.targets for row in rows]).numpy().astype(np.float32, copy=False)
    masks = torch.stack([row.mask for row in rows]).numpy().astype(np.float32, copy=False)
    if features.shape != (len(rows), INPUT_DIM) or targets.shape != (len(rows), 2):
        raise ValueError("hybrid rows have inconsistent numeric shapes")
    if masks.shape != targets.shape:
        raise ValueError("hybrid masks must match target shape")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("hybrid numeric arrays must be finite")
    if not np.isin(masks, (0.0, 1.0)).all():
        raise ValueError("hybrid masks must contain only zero and one")

    max_deal_id = max(len(row.deal_id) for row in rows)
    return HybridArrays(
        features=features,
        targets=targets,
        masks=masks,
        centers=np.asarray([int(row.center) for row in rows], dtype=np.int8),
        baseline_margins=np.asarray(
            [row.baseline_margin for row in rows], dtype=np.float64
        ),
        shuffle_seeds=np.asarray([row.shuffle_seed for row in rows], dtype=np.int64),
        room_ids=np.asarray([row.room_id for row in rows], dtype=np.int8),
        physical_seats=np.asarray([row.physical_seat for row in rows], dtype=np.int8),
        bid_indices=np.asarray([row.bid_index for row in rows], dtype=np.int8),
        deal_ids=np.asarray(
            [row.deal_id for row in rows], dtype=f"<U{max(1, max_deal_id)}"
        ),
    )


def save_hybrid_npz(destination: Path, deals: Sequence[HybridDeal]) -> HybridArrays:
    """Write one simple compressed NPZ artifact with no Python objects."""

    destination = Path(destination)
    if destination.suffix != ".npz":
        raise ValueError("hybrid dataset path must end in .npz")
    arrays = stack_hybrid_deals(deals)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{name: getattr(arrays, name) for name in arrays.array_names()},
    )
    return arrays


def load_hybrid_npz(source: Path) -> HybridArrays:
    """Load the minimal dataset with NumPy object deserialization disabled."""

    source = Path(source)
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAY_NAMES):
            raise ValueError("hybrid dataset has an unexpected array schema")
        values = {name: archive[name].copy() for name in _ARRAY_NAMES}
    arrays = HybridArrays(**values)
    row_count = arrays.features.shape[0]
    if arrays.features.shape != (row_count, INPUT_DIM):
        raise ValueError("hybrid features must have shape (N, 167)")
    if arrays.targets.shape != (row_count, 2) or arrays.masks.shape != (row_count, 2):
        raise ValueError("hybrid targets and masks must have shape (N, 2)")
    if any(getattr(arrays, name).shape != (row_count,) for name in _ARRAY_NAMES[3:]):
        raise ValueError("hybrid metadata arrays must have shape (N,)")
    if any(getattr(arrays, name).dtype == object for name in _ARRAY_NAMES):
        raise ValueError("hybrid dataset must not contain object arrays")
    return arrays
