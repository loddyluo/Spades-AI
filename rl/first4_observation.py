"""Leakage-safe 536-dimensional observation for first-four Spades play.

The encoder is intentionally separate from the legacy 264-dimensional
``RLFeatureEncoder``.  It accepts only :class:`FirstFourObservation`, which
contains the acting player's hand and public information, never the other
three concealed hands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from trick_taking.card import Card, Rank, Suit
from trick_taking.game_state import GameState, Phase


ENCODER_SCHEMA = "first4-observation-v2-536"
MAX_HISTORY_CARDS = 15


@dataclass(frozen=True, slots=True)
class ObservedPlay:
    """One publicly observed play in relative-seat coordinates."""

    card_id: int
    rank: int
    suit: int
    relative_seat: int

    def __post_init__(self) -> None:
        if not 0 <= self.card_id < 52:
            raise ValueError("card_id must be in [0, 51]")
        if not 2 <= self.rank <= 14:
            raise ValueError("rank must be in [2, 14]")
        if not 0 <= self.suit < 4:
            raise ValueError("suit must be in [0, 3]")
        if not 0 <= self.relative_seat < 4:
            raise ValueError("relative_seat must be in [0, 3]")
        if self.card_id != self.suit * 13 + (self.rank - 2):
            raise ValueError("card_id must agree with rank and suit")


@dataclass(frozen=True, slots=True)
class FirstFourObservation:
    """Complete information exposed to the first-four actor and critic.

    Relative seats are always ordered as ``self, upper, partner, lower``.
    The object deliberately stores card ids rather than a ``GameState`` so a
    model cannot accidentally inspect concealed opponent hands.
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
        if any(not 1 <= value <= 13 for value in self.bids):
            raise ValueError("first-four-v2 supports only numeric bids 1 through 13")
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
        if len(self.public_voids) != 3 or any(len(row) != 4 for row in self.public_voids):
            raise ValueError("public_voids must have shape (3, 4)")
        if any(not math.isfinite(value) for value in self.contract_progress):
            raise ValueError("contract_progress must be finite")


def relative_seat(player_id: int, absolute_seat: int) -> int:
    """Map an absolute seat to ``self, upper, partner, lower`` coordinates."""

    if not 0 <= player_id < 4 or not 0 <= absolute_seat < 4:
        raise ValueError("seat ids must be in [0, 3]")
    return (player_id - absolute_seat) % 4


def _numeric_bid(value: object) -> int:
    if isinstance(value, str) and value.startswith("bid_"):
        try:
            parsed = int(value.split("_", 1)[1])
        except ValueError as error:
            raise ValueError(f"invalid numeric bid {value!r}") from error
        if 1 <= parsed <= 13:
            return parsed
    if type(value) is int and 1 <= value <= 13:
        return value
    raise ValueError(f"first-four-v2 requires a non-Nil numeric bid, got {value!r}")


def _current_winner(table_cards: Sequence[tuple[int, Card]]) -> int | None:
    if not table_cards:
        return None
    lead_suit = table_cards[0][1].suit
    best_seat, best_card = table_cards[0]
    for seat, card in table_cards[1:]:
        if card.suit == Suit.SPADES:
            if best_card.suit != Suit.SPADES or card.rank.value > best_card.rank.value:
                best_seat, best_card = seat, card
        elif best_card.suit != Suit.SPADES and card.suit == lead_suit:
            if best_card.suit != lead_suit or card.rank.value > best_card.rank.value:
                best_seat, best_card = seat, card
    return best_seat


def _public_voids(state: GameState, player_id: int) -> tuple[
    tuple[bool, bool, bool, bool],
    tuple[bool, bool, bool, bool],
    tuple[bool, bool, bool, bool],
]:
    flags = [[False] * 4 for _ in range(4)]
    tricks: list[Sequence[tuple[int, Card]]] = [record.cards for record in state.trick_history]
    if state.table_cards:
        tricks.append(state.table_cards)
    for cards in tricks:
        if not cards:
            continue
        lead_suit = cards[0][1].suit.value
        for seat, card in cards[1:]:
            if card.suit.value != lead_suit:
                flags[relative_seat(player_id, seat)][lead_suit] = True
    return tuple(tuple(flags[rel]) for rel in (1, 2, 3))  # type: ignore[return-value]


def build_first_four_observation(
    state: GameState,
    player_id: int,
    legal_cards: Sequence[Card],
) -> FirstFourObservation:
    """Build a leakage-safe observation from one acting player's perspective."""

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
        raise ValueError("the v2 observation is only valid before the fourth trick completes")
    if len(state.table_cards) > 3:
        raise ValueError("an acting observation cannot have a complete table")
    if any(seat == player_id for seat, _ in state.table_cards):
        raise ValueError("the acting player cannot already have played in the current trick")

    absolute_by_relative = tuple((player_id - rel) % 4 for rel in range(4))
    bids = tuple(_numeric_bid(state.max_bid[seat]) for seat in absolute_by_relative)
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

    own_team = state.teams[player_id]
    if tuple(state.teams) != (0, 1, 0, 1):
        raise ValueError("Spades state must use fixed teams 0/2 versus 1/3")
    opponent_team = 1 - own_team
    team_bids = [0, 0]
    team_tricks = [0, 0]
    for seat in range(4):
        team = int(state.teams[seat])
        team_bids[team] += _numeric_bid(state.max_bid[seat])
        team_tricks[team] += int(state.tricks_won[seat])
    contract_progress = (
        (team_tricks[own_team] - team_bids[own_team]) / 26.0,
        (team_tricks[opponent_team] - team_bids[opponent_team]) / 26.0,
    )

    winner = _current_winner(state.table_cards)
    return FirstFourObservation(
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


class FirstFourFeatureEncoderV2:
    """Encode :class:`FirstFourObservation` into the fixed 536-vector."""

    SCHEMA = ENCODER_SCHEMA
    TOTAL_DIM = 536

    HAND_START = 0
    LEGAL_START = 52
    BIDS_START = 104
    HISTORY_START = 156
    HISTORY_SLOT_DIM = 21
    CONTEXT_START = 471
    TRICKS_WON_START = 493
    CONTRACT_START = 509
    VOIDS_START = 511
    SPADES_BROKEN_INDEX = 523
    HAND_DERIVED_START = 524

    def encode(self, observation: FirstFourObservation) -> np.ndarray:
        if not isinstance(observation, FirstFourObservation):
            raise TypeError("encoder accepts only FirstFourObservation")
        feature = np.zeros(self.TOTAL_DIM, dtype=np.float32)

        feature[np.asarray(observation.hand_card_ids, dtype=np.int64)] = 1.0
        legal_indices = self.LEGAL_START + np.asarray(
            observation.legal_card_ids, dtype=np.int64
        )
        feature[legal_indices] = 1.0

        for rel, bid in enumerate(observation.bids):
            feature[self.BIDS_START + rel * 13 + (bid - 1)] = 1.0

        for slot, play in enumerate(reversed(observation.history)):
            start = self.HISTORY_START + slot * self.HISTORY_SLOT_DIM
            feature[start + (play.rank - 2)] = 1.0
            feature[start + 13 + play.suit] = 1.0
            feature[start + 17 + play.relative_seat] = 1.0

        context = self.CONTEXT_START
        feature[context + observation.tricks_played] = 1.0
        feature[context + 4 + (observation.trick_position - 1)] = 1.0
        feature[context + 8 + observation.trick_leader_relative] = 1.0
        lead_index = 4 if observation.lead_suit is None else observation.lead_suit
        feature[context + 12 + lead_index] = 1.0
        winner_index = (
            4
            if observation.current_winner_relative is None
            else observation.current_winner_relative
        )
        feature[context + 17 + winner_index] = 1.0

        for rel, count in enumerate(observation.tricks_won):
            feature[self.TRICKS_WON_START + rel * 4 + count] = 1.0
        feature[self.CONTRACT_START : self.CONTRACT_START + 2] = np.asarray(
            observation.contract_progress, dtype=np.float32
        )
        for other_index, row in enumerate(observation.public_voids):
            for suit, is_void in enumerate(row):
                feature[self.VOIDS_START + other_index * 4 + suit] = float(is_void)
        feature[self.SPADES_BROKEN_INDEX] = float(observation.spades_broken)

        hand_cards = [
            Card(Suit(card_id // 13), Rank((card_id % 13) + 2))
            for card_id in observation.hand_card_ids
        ]
        for suit in Suit:
            cards = [card for card in hand_cards if card.suit == suit]
            feature[self.HAND_DERIVED_START + suit.value] = len(cards) / 13.0
            high_count = sum(card.rank in (Rank.QUEEN, Rank.KING, Rank.ACE) for card in cards)
            feature[self.HAND_DERIVED_START + 4 + suit.value] = high_count / 3.0
            feature[self.HAND_DERIVED_START + 8 + suit.value] = float(
                any(card.rank == Rank.ACE for card in cards)
            )

        if feature.shape != (self.TOTAL_DIM,) or not np.isfinite(feature).all():
            raise AssertionError("first-four-v2 encoder produced an invalid feature vector")
        return feature

    @classmethod
    def segment_ranges(cls) -> dict[str, tuple[int, int]]:
        return {
            "hand": (cls.HAND_START, cls.LEGAL_START),
            "legal": (cls.LEGAL_START, cls.BIDS_START),
            "bids": (cls.BIDS_START, cls.HISTORY_START),
            "history": (cls.HISTORY_START, cls.CONTEXT_START),
            "context": (cls.CONTEXT_START, cls.TRICKS_WON_START),
            "tricks_won": (cls.TRICKS_WON_START, cls.CONTRACT_START),
            "contract": (cls.CONTRACT_START, cls.VOIDS_START),
            "voids": (cls.VOIDS_START, cls.SPADES_BROKEN_INDEX),
            "spades_broken": (cls.SPADES_BROKEN_INDEX, cls.HAND_DERIVED_START),
            "hand_derived": (cls.HAND_DERIVED_START, cls.TOTAL_DIM),
        }


__all__ = [
    "ENCODER_SCHEMA",
    "MAX_HISTORY_CARDS",
    "FirstFourFeatureEncoderV2",
    "FirstFourObservation",
    "ObservedPlay",
    "build_first_four_observation",
    "relative_seat",
]
