"""Exact automatic-showdown checks for authoritative Spades game states.

This module is deliberately separate from the acting-player path.  It accepts
all four real hands only to prove that every legal continuation has the same
scoring-relevant result, then builds one deterministic legal continuation for
the existing settlement and replay code to consume after confirmation.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from trick_taking.card import Card, Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.game_state import GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


ShowdownStatus = Literal["fixed", "variable", "timeout"]


class ShowdownStateError(ValueError):
    """The supplied state is not a valid complete-trick showdown boundary."""


@dataclass(frozen=True)
class ShowdownPlay:
    seat: int
    card: Card

    def to_payload(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "card": f"{self.card.rank.short}{self.card.suit.short}",
        }


@dataclass(frozen=True)
class ShowdownResolution:
    team_tricks: tuple[int, int]
    nil_outcomes: tuple[bool | None, bool | None, bool | None, bool | None]
    continuation: tuple[ShowdownPlay, ...]
    final_tricks_won: tuple[int, int, int, int]

    def to_payload(self) -> dict[str, Any]:
        return {
            "teamTricks": list(self.team_tricks),
            "nilOutcomes": list(self.nil_outcomes),
            "continuation": [play.to_payload() for play in self.continuation],
            "finalTricksWon": list(self.final_tricks_won),
        }


@dataclass(frozen=True)
class ShowdownCheck:
    status: ShowdownStatus
    resolution: ShowdownResolution | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.resolution is not None:
            payload["resolution"] = self.resolution.to_payload()
        return payload


def outcome_signature(state: GameState) -> tuple[int, int]:
    """Return the terminal data that can affect Spades scoring."""
    team0 = sum(
        state.tricks_won[seat]
        for seat in range(4)
        if state.teams[seat] == 0
    )
    nil_mask = sum(
        1 << seat
        for seat, bid in enumerate(state.max_bid)
        if bid in ("nil", "blind_nil") and state.tricks_won[seat] > 0
    )
    return team0, nil_mask


def _valid_bid(bid: Any) -> bool:
    if bid in ("nil", "blind_nil"):
        return True
    if isinstance(bid, int) and not isinstance(bid, bool):
        return 1 <= bid <= 13
    if isinstance(bid, str) and bid.startswith("bid_"):
        try:
            value = int(bid.removeprefix("bid_"))
        except ValueError:
            return False
        return 1 <= value <= 13
    return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ShowdownStateError(message)


def _validate_showdown_state(state: GameState) -> None:
    """Validate and replay all public history before exposing hidden hands."""
    _require(state.num_players == 4, "showdown requires four players")
    _require(state.phase == Phase.PLAYING, "showdown requires the playing phase")
    _require(not state.table_cards, "showdown is only checked between tricks")
    _require(len(state.hands) == 4, "showdown requires four complete hands")

    hand_sizes = [len(hand) for hand in state.hands]
    _require(
        len(set(hand_sizes)) == 1 and 1 <= hand_sizes[0] <= 5,
        "showdown requires one to five equally sized hands",
    )
    remaining_tricks = hand_sizes[0]
    completed_tricks = 13 - remaining_tricks

    _require(
        state.tricks_played == completed_tricks,
        "tricks_played does not match the remaining hands",
    )
    _require(
        len(state.trick_history) == completed_tricks,
        "trick history length does not match tricks_played",
    )
    _require(state.teams == [0, 1, 0, 1], "unexpected Spades teams")
    _require(
        len(state.max_bid) == 4 and all(_valid_bid(bid) for bid in state.max_bid),
        "all four bids must be complete",
    )
    _require(
        len(state.tricks_won) == 4
        and all(isinstance(value, int) and value >= 0 for value in state.tricks_won)
        and sum(state.tricks_won) == completed_tricks,
        "trick totals do not match tricks_played",
    )
    _require(
        0 <= state.turn < 4
        and 0 <= state.trick_leader < 4
        and state.turn == state.trick_leader,
        "turn and leader must agree between tricks",
    )
    _require(state.trump_suit == Suit.SPADES, "Spades must be the trump suit")
    _require(
        state.trump_broken == state.spades_broken,
        "Spades broken flags disagree",
    )

    original_hands = [list(hand) for hand in state.hands]
    history_cards: list[Card] = []
    for trick in state.trick_history:
        _require(len(trick.cards) == 4, "every completed trick must have four cards")
        _require(
            0 <= trick.leader < 4 and 0 <= trick.winner < 4,
            "trick contains an invalid seat",
        )
        for seat, card in trick.cards:
            _require(0 <= seat < 4, "trick contains an invalid card seat")
            _require(isinstance(card, Card), "trick contains an invalid card")
            original_hands[seat].append(card)
            history_cards.append(card)

    remaining_cards = [card for hand in state.hands for card in hand]
    all_known_cards = remaining_cards + history_cards
    _require(
        len(all_known_cards) == 52
        and len({card.card_id for card in all_known_cards}) == 52
        and {card.card_id for card in all_known_cards} == set(range(52)),
        "remaining hands and history must form one unique standard deck",
    )
    _require(
        all(len(hand) == 13 for hand in original_hands),
        "each seat must account for exactly thirteen cards",
    )

    expected_played = cards_to_bitset(history_cards)
    _require(
        state.played_bitset == expected_played,
        "played bitset does not match trick history",
    )
    _require(
        len(state.hand_bitsets) == 4
        and all(
            state.hand_bitsets[seat] == cards_to_bitset(state.hands[seat])
            for seat in range(4)
        ),
        "hand bitsets do not match remaining hands",
    )
    if state.all_cards:
        _require(
            len(state.all_cards) == 52
            and {card.card_id for card in state.all_cards} == set(range(52)),
            "all_cards is not a standard deck",
        )

    # Reconstruct the deal and replay every card.  This catches incorrect
    # leaders, winners, turn order, follow-suit violations, and trick totals.
    rules = SpadesRules()
    replay = GameState()
    replay.num_players = 4
    replay.phase = Phase.PLAYING
    replay.hands = [list(hand) for hand in original_hands]
    replay.hand_bitsets = [cards_to_bitset(hand) for hand in replay.hands]
    replay.all_cards = list(_STANDARD_CARDS)
    replay.max_bid = list(state.max_bid)
    replay.teams = [0, 1, 0, 1]
    replay.table_cards = []
    replay.trump_suit = Suit.SPADES
    replay.trump_broken = False
    replay.spades_broken = False
    replay.tricks_won = [0, 0, 0, 0]
    replay.cards_won = [[] for _ in range(4)]
    replay.trick_history = []
    replay.played_bitset = 0
    replay.tricks_played = 0

    if state.trick_history:
        replay.turn = replay.trick_leader = state.trick_history[0].leader

    for trick_index, recorded in enumerate(state.trick_history):
        _require(
            recorded.leader == replay.trick_leader,
            f"trick {trick_index} leader does not follow the prior winner",
        )
        expected_seats = [
            (recorded.leader + offset) % 4
            for offset in range(4)
        ]
        _require(
            [seat for seat, _ in recorded.cards] == expected_seats,
            f"trick {trick_index} card order is invalid",
        )
        for seat, card in recorded.cards:
            _require(card in replay.hands[seat], f"seat {seat} did not hold {card}")
            legal = rules.playable(replay, replay.hands[seat], seat)
            _require(card in legal, f"seat {seat} played an illegal card")
            replay.play_card_to_table(seat, card)
            if card.suit == Suit.SPADES:
                replay.trump_broken = replay.spades_broken = True
            replay.turn = (seat + 1) % 4

        winner = rules.winner_trick(replay)
        _require(
            winner == recorded.winner,
            f"trick {trick_index} recorded the wrong winner",
        )
        replay.complete_trick(winner)
        replay.turn = replay.trick_leader = winner

    _require(
        all(
            set(replay.hands[seat]) == set(state.hands[seat])
            for seat in range(4)
        ),
        "replayed history does not produce the supplied remaining hands",
    )
    _require(replay.tricks_won == state.tricks_won, "trick totals disagree with history")
    _require(
        replay.turn == state.turn and replay.trick_leader == state.trick_leader,
        "current leader does not match the last trick winner",
    )
    _require(
        replay.spades_broken == state.spades_broken,
        "Spades broken state does not match history",
    )


def validate_showdown_state(state: GameState) -> None:
    """Public validation entry point for authoritative state builders."""
    _validate_showdown_state(state)


def _prepare_completion_copy(state: GameState) -> GameState:
    resolved = copy.deepcopy(state)
    # Some state builders do not materialize cards_won because Spades scoring
    # only needs trick counts.  Rebuild it so GameState.complete_trick remains
    # safe and the terminal state is internally complete.
    resolved.cards_won = [[] for _ in range(4)]
    for trick in resolved.trick_history:
        resolved.cards_won[trick.winner].extend(card for _, card in trick.cards)
    return resolved


def _apply_one_play(
    state: GameState,
    play: ShowdownPlay,
    rules: SpadesRules,
) -> None:
    if state.tricks_played >= 13:
        raise ValueError("showdown continuation contains plays after the hand ended")
    if play.seat != state.turn:
        raise ValueError(
            f"showdown continuation seat mismatch: expected {state.turn}, got {play.seat}"
        )
    if play.card not in state.hands[play.seat]:
        raise ValueError("showdown continuation card is not in the player's hand")
    legal = rules.playable(state, state.hands[play.seat], play.seat)
    if play.card not in legal:
        raise ValueError("showdown continuation contains an illegal card")

    state.play_card_to_table(play.seat, play.card)
    if play.card.suit == Suit.SPADES:
        state.trump_broken = state.spades_broken = True
    state.turn = (play.seat + 1) % state.num_players
    if state.trick_complete:
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.turn = state.trick_leader = winner


def deterministic_continuation(
    state: GameState,
) -> tuple[tuple[ShowdownPlay, ...], GameState]:
    """Complete a validated state by always choosing the lowest legal card ID."""
    resolved = _prepare_completion_copy(state)
    rules = SpadesRules()
    plays: list[ShowdownPlay] = []

    while not rules.end_trickgame(resolved):
        seat = resolved.turn
        legal = rules.playable(resolved, resolved.hands[seat], seat)
        if not legal:
            raise RuntimeError("validated showdown state has no legal continuation")
        play = ShowdownPlay(seat=seat, card=min(legal, key=lambda card: card.card_id))
        plays.append(play)
        _apply_one_play(resolved, play, rules)

    return tuple(plays), resolved


def apply_showdown_continuation(
    state: GameState,
    continuation: Sequence[ShowdownPlay],
) -> GameState:
    """Revalidate and apply a stored continuation to a deep copy of ``state``."""
    _validate_showdown_state(state)
    resolved = _prepare_completion_copy(state)
    rules = SpadesRules()
    expected_plays = sum(len(hand) for hand in state.hands)
    if len(continuation) != expected_plays:
        raise ValueError(
            f"showdown continuation has {len(continuation)} plays; expected {expected_plays}"
        )
    for play in continuation:
        if not isinstance(play, ShowdownPlay):
            raise ValueError("showdown continuation contains an invalid play")
        _apply_one_play(resolved, play, rules)

    if (
        resolved.tricks_played != 13
        or resolved.table_cards
        or any(resolved.hands)
    ):
        raise ValueError("showdown continuation did not produce a complete hand")
    return resolved


def _nil_outcomes(
    state: GameState,
) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    outcomes: list[bool | None] = []
    for seat, bid in enumerate(state.max_bid):
        if bid in ("nil", "blind_nil"):
            outcomes.append(state.tricks_won[seat] == 0)
        else:
            outcomes.append(None)
    return tuple(outcomes)  # type: ignore[return-value]


def check_for_showdown(
    state: GameState,
    solver: ExactDoubleDummyCppFastestSolver | Any | None = None,
    *,
    time_budget_seconds: float = 1.0,
) -> ShowdownCheck:
    """Prove a fixed outcome and prepare a settlement line when one exists."""
    started = time.monotonic()
    _validate_showdown_state(state)
    if solver is None:
        solver = ExactDoubleDummyCppFastestSolver()

    budget = max(0.0, float(time_budget_seconds))
    remaining = budget - (time.monotonic() - started)
    if remaining <= 0.0:
        return ShowdownCheck(status="timeout")

    raw = solver.analyze_forced_outcome(
        state,
        time_budget_seconds=remaining,
    )
    status = raw.get("status")
    if status in ("variable", "timeout"):
        return ShowdownCheck(status=status)
    if status != "fixed":
        raise RuntimeError(f"unknown forced-outcome status: {status!r}")

    continuation, terminal = deterministic_continuation(state)
    proven_signature = (
        int(raw["team0_final_tricks"]),
        int(raw["nil_broken_mask"]),
    )
    generated_signature = outcome_signature(terminal)
    if generated_signature != proven_signature:
        raise RuntimeError(
            "forced-outcome signature mismatch: "
            f"native={proven_signature}, generated={generated_signature}"
        )

    team_tricks = (
        proven_signature[0],
        13 - proven_signature[0],
    )
    resolution = ShowdownResolution(
        team_tricks=team_tricks,
        nil_outcomes=_nil_outcomes(terminal),
        continuation=continuation,
        final_tricks_won=tuple(terminal.tricks_won),  # type: ignore[arg-type]
    )
    return ShowdownCheck(status="fixed", resolution=resolution)


__all__ = [
    "ShowdownCheck",
    "ShowdownPlay",
    "ShowdownResolution",
    "ShowdownStateError",
    "apply_showdown_continuation",
    "check_for_showdown",
    "deterministic_continuation",
    "outcome_signature",
    "validate_showdown_state",
]
