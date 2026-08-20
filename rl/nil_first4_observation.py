"""Leakage-safe first-four observation for Spades deals containing Nil bids.

The layout deliberately stays 536-dimensional so the four Nil-role policies can
be initialized from the deployed non-Nil solver-leaf trainer.  A Nil bid is
encoded as an all-zero 13-way bid block for that relative seat; numeric bids use
the same one-hot encoding as :mod:`rl.first4_observation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rl.first4_observation import (
    MAX_HISTORY_CARDS,
    FirstFourFeatureEncoderV2,
    FirstFourObservation,
    ObservedPlay,
    _current_winner,
    _public_voids,
    relative_seat,
)
from trick_taking.card import Card
from trick_taking.game_state import GameState, Phase


NIL_ENCODER_SCHEMA = "first4-nil-observation-v1-536"


def nil_numeric_bid(value: object) -> int:
    """Return 0 for Nil and 1..13 for a numeric bid."""

    if value in ("nil", "blind_nil"):
        return 0
    if isinstance(value, str) and value.startswith("bid_"):
        try:
            parsed = int(value.split("_", 1)[1])
        except ValueError as error:
            raise ValueError(f"invalid numeric bid {value!r}") from error
        if 1 <= parsed <= 13:
            return parsed
    if type(value) is int and 1 <= value <= 13:
        return value
    raise ValueError(f"Nil first-four observation got unsupported bid {value!r}")


@dataclass(frozen=True, slots=True)
class NilFirstFourObservation:
    """Public observation for a deal containing one or more Nil bidders.

    Relative seats use the same ``self, upper, partner, lower`` convention as
    :class:`rl.first4_observation.FirstFourObservation`.
    """

    player_id: int
    hand_card_ids: tuple[int, ...]
    legal_card_ids: tuple[int, ...]
    bids: tuple[int, int, int, int]
    history: tuple[ObservedPlay, ...]
    tricks_played: int
    trick_position: int
    trick_leader_relative: int
    lead_suit: int | None
    current_winner_relative: int | None
    tricks_won: tuple[int, int, int, int]
    contract_progress: tuple[float, float]
    public_voids: tuple[
        tuple[bool, bool, bool, bool],
        tuple[bool, bool, bool, bool],
        tuple[bool, bool, bool, bool],
    ]
    spades_broken: bool

    def __post_init__(self) -> None:
        if not 0 <= self.player_id < 4:
            raise ValueError("player_id must be in [0, 3]")
        if len(set(self.hand_card_ids)) != len(self.hand_card_ids):
            raise ValueError("hand_card_ids must be unique")
        if any(not 0 <= value < 52 for value in self.hand_card_ids):
            raise ValueError("hand_card_ids must be in [0, 51]")
        if not self.legal_card_ids:
            raise ValueError("legal_card_ids must be nonempty")
        if len(set(self.legal_card_ids)) != len(self.legal_card_ids):
            raise ValueError("legal_card_ids must be unique")
        if not set(self.legal_card_ids).issubset(self.hand_card_ids):
            raise ValueError("legal cards must be a subset of the acting hand")
        if any(not 0 <= value <= 13 for value in self.bids):
            raise ValueError("Nil first-four bids must be in [0, 13]")
        if self.bids.count(0) < 1:
            raise ValueError("Nil first-four observation requires at least one Nil bid")
        if len(self.history) > MAX_HISTORY_CARDS:
            raise ValueError("first-four history cannot contain more than 15 cards")
        if not 0 <= self.tricks_played < 4:
            raise ValueError("tricks_played must be in [0, 3]")
        if not 1 <= self.trick_position <= 4:
            raise ValueError("trick_position must be in [1, 4]")
        if not 0 <= self.trick_leader_relative < 4:
            raise ValueError("trick_leader_relative must be in [0, 3]")
        if self.lead_suit is not None and not 0 <= self.lead_suit < 4:
            raise ValueError("lead_suit must be None or in [0, 3]")
        if self.current_winner_relative is not None and not (
            0 <= self.current_winner_relative < 4
        ):
            raise ValueError("current_winner_relative must be None or in [0, 3]")
        if any(not 0 <= value <= 3 for value in self.tricks_won):
            raise ValueError("pre-leaf tricks_won values must be in [0, 3]")
        if len(self.contract_progress) != 2 or not np.isfinite(
            np.asarray(self.contract_progress, dtype=np.float64)
        ).all():
            raise ValueError("contract_progress must contain two finite values")
        if len(self.public_voids) != 3 or any(len(row) != 4 for row in self.public_voids):
            raise ValueError("public_voids must have shape (3, 4)")


def build_nil_first_four_observation(
    state: GameState,
    player_id: int,
    legal_cards: Sequence[Card],
) -> NilFirstFourObservation:
    """Build a one-or-more-Nil observation without concealed-card leakage.

    The four-role policies were trained on exactly-one-Nil deals. Runtime
    multi-Nil play uses the compatible representation in which every Nil bid
    has an all-zero 13-way bid block.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if state.phase is not Phase.PLAYING:
        raise ValueError("first-four observations require Phase.PLAYING")
    if state.num_players != 4 or len(state.hands) != 4:
        raise ValueError("first-four observations require four players")
    if not 0 <= player_id < 4 or state.turn != player_id:
        raise ValueError("observation must be built for the acting player")
    if len(state.max_bid) != 4 or len(state.teams) != 4 or len(state.tricks_won) != 4:
        raise ValueError("bids, teams, and trick counts must cover all four seats")
    if not legal_cards:
        raise ValueError("legal_cards must be nonempty")
    if any(card not in state.hands[player_id] for card in legal_cards):
        raise ValueError("legal_cards must come from the acting hand")
    if state.tricks_played >= 4:
        raise ValueError("the Nil observation is only valid before trick four completes")
    if len(state.table_cards) > 3:
        raise ValueError("an acting observation cannot have a complete table")
    if any(seat == player_id for seat, _ in state.table_cards):
        raise ValueError("the acting player cannot already have played in this trick")
    if tuple(state.teams) != (0, 1, 0, 1):
        raise ValueError("Spades state must use fixed teams 0/2 versus 1/3")

    absolute_by_relative = tuple((player_id - rel) % 4 for rel in range(4))
    bids = tuple(nil_numeric_bid(state.max_bid[seat]) for seat in absolute_by_relative)
    if bids.count(0) < 1:
        raise ValueError("Nil observation requires at least one Nil bidder")
    tricks_won = tuple(int(state.tricks_won[seat]) for seat in absolute_by_relative)

    public_cards: list[tuple[int, Card]] = []
    for record in state.trick_history:
        public_cards.extend(record.cards)
    public_cards.extend(state.table_cards)
    expected_public = state.tricks_played * 4 + len(state.table_cards)
    if len(public_cards) != expected_public or len(public_cards) > MAX_HISTORY_CARDS:
        raise ValueError("public trick history is inconsistent with the current state")
    history = tuple(
        ObservedPlay(
            card_id=card.card_id,
            rank=card.rank.value,
            suit=card.suit.value,
            relative_seat=relative_seat(player_id, seat),
        )
        for seat, card in public_cards
    )

    own_team = int(state.teams[player_id])
    opponent_team = 1 - own_team
    team_bids = [0, 0]
    team_tricks = [0, 0]
    for seat in range(4):
        team = int(state.teams[seat])
        team_bids[team] += nil_numeric_bid(state.max_bid[seat])
        team_tricks[team] += int(state.tricks_won[seat])
    contract_progress = (
        (team_tricks[own_team] - team_bids[own_team]) / 26.0,
        (team_tricks[opponent_team] - team_bids[opponent_team]) / 26.0,
    )

    winner = _current_winner(state.table_cards)
    return NilFirstFourObservation(
        player_id=player_id,
        hand_card_ids=tuple(sorted(card.card_id for card in state.hands[player_id])),
        legal_card_ids=tuple(sorted(card.card_id for card in legal_cards)),
        bids=bids,  # type: ignore[arg-type]
        history=history,
        tricks_played=int(state.tricks_played),
        trick_position=len(state.table_cards) + 1,
        trick_leader_relative=relative_seat(player_id, state.trick_leader),
        lead_suit=None if state.lead_suit is None else state.lead_suit.value,
        current_winner_relative=None if winner is None else relative_seat(player_id, winner),
        tricks_won=tricks_won,  # type: ignore[arg-type]
        contract_progress=contract_progress,
        public_voids=_public_voids(state, player_id),
        spades_broken=bool(state.spades_broken or state.trump_broken),
    )


class NilFirstFourFeatureEncoderV1(FirstFourFeatureEncoderV2):
    """Encode one-or-more-Nil observations in the compatible 536 layout."""

    SCHEMA = NIL_ENCODER_SCHEMA

    def encode(self, observation: NilFirstFourObservation) -> np.ndarray:
        if not isinstance(observation, NilFirstFourObservation):
            raise TypeError("encoder accepts only NilFirstFourObservation")
        surrogate = FirstFourObservation(
            player_id=observation.player_id,
            hand_card_ids=observation.hand_card_ids,
            legal_card_ids=observation.legal_card_ids,
            bids=tuple(max(1, value) for value in observation.bids),  # type: ignore[arg-type]
            history=observation.history,
            tricks_played=observation.tricks_played,
            trick_position=observation.trick_position,
            trick_leader_relative=observation.trick_leader_relative,
            lead_suit=observation.lead_suit,
            current_winner_relative=observation.current_winner_relative,
            tricks_won=observation.tricks_won,
            contract_progress=observation.contract_progress,
            public_voids=observation.public_voids,
            spades_broken=observation.spades_broken,
        )
        feature = super().encode(surrogate)
        for relative, bid in enumerate(observation.bids):
            if bid == 0:
                start = self.BIDS_START + relative * 13
                feature[start : start + 13] = 0.0
        if feature.shape != (self.TOTAL_DIM,) or not np.isfinite(feature).all():
            raise AssertionError("Nil first-four encoder produced an invalid feature vector")
        return feature


__all__ = [
    "NIL_ENCODER_SCHEMA",
    "NilFirstFourFeatureEncoderV1",
    "NilFirstFourObservation",
    "build_nil_first_four_observation",
    "nil_numeric_bid",
]
