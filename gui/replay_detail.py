"""Shared detailed-replay helpers for the HTTP and WebSocket GUI servers."""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Any, Mapping, Sequence

from trick_taking.card import Card, Suit, _STANDARD_CARDS, cards_to_bitset
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import (
    expand_equivalent_root_q_values,
)


REPLAY_ANALYSIS_FIRST_PLAY_INDEX = 12  # zero-based: first action of trick 4
REPLAY_ANALYSIS_LAST_PLAY_INDEX = 51


def card_to_code(card: Card) -> str:
    return f"{card.rank.short}{card.suit.short}"


def _json_safe(value: Any) -> Any:
    """Convert player diagnostics to a compact JSON-safe structure."""
    if isinstance(value, Card):
        return card_to_code(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def serialize_ai_play_info(
    play_info: Mapping[str, Any] | None,
    *,
    chosen_card: Card,
    seat: int,
) -> dict[str, Any]:
    """Serialize the exact diagnostics captured at decision time."""
    safe_info = _json_safe(dict(play_info or {}))
    if not isinstance(safe_info, dict):  # defensive; Mapping above guarantees this
        safe_info = {}
    return {
        "schema_version": 1,
        "seat": int(seat),
        "chosen_card": card_to_code(chosen_card),
        **safe_info,
    }


def _valid_bid(value: Any) -> bool:
    if value in ("nil", "blind_nil"):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return 1 <= value <= 13
    if isinstance(value, str) and value.startswith("bid_"):
        try:
            return 1 <= int(value.removeprefix("bid_")) <= 13
        except ValueError:
            return False
    return False


def build_replay_solver_state(
    initial_hands: Sequence[Sequence[Card]],
    max_bid: Sequence[Any],
    plays: Sequence[tuple[int, Card]],
    *,
    first_leader: int,
    play_index: int,
) -> tuple[GameState, tuple[int, Card]]:
    """Rebuild the full-information state immediately before one replay play."""
    if not REPLAY_ANALYSIS_FIRST_PLAY_INDEX <= play_index <= REPLAY_ANALYSIS_LAST_PLAY_INDEX:
        raise ValueError("上帝视角只计算第 4 至第 13 墩（后十墩）")
    if len(initial_hands) != 4 or any(len(hand) != 13 for hand in initial_hands):
        raise ValueError("initialHands 必须包含四家各 13 张牌")

    hands = [list(hand) for hand in initial_hands]
    all_cards = [card for hand in hands for card in hand]
    if (
        len(all_cards) != 52
        or {card.card_id for card in all_cards} != set(range(52))
    ):
        raise ValueError("initialHands 必须无重复地组成标准 52 张牌")
    if len(max_bid) != 4 or not all(_valid_bid(value) for value in max_bid):
        raise ValueError("bids 必须包含四个有效叫牌")
    if not 0 <= first_leader < 4:
        raise ValueError("firstLeader 必须是 0-3 的座位编号")
    if len(plays) != 52:
        raise ValueError("plays 必须包含完整的 52 次出牌")
    if play_index >= len(plays):
        raise ValueError("playIndex 超出出牌记录范围")

    state = GameState()
    state.init_for_deal(
        4,
        hands,
        [],
        list(_STANDARD_CARDS),
    )
    state.phase = Phase.PLAYING
    state.max_bid = list(max_bid)
    state.bids = [
        Bid(player_id=seat, value=value, is_pass=False)
        for seat, value in enumerate(max_bid)
    ]
    state.teams = [0, 1, 0, 1]
    state.trump_suit = Suit.SPADES
    state.turn = first_leader
    state.current_bidder = first_leader
    state.trick_leader = first_leader
    rules = SpadesRules()

    for history_index, (seat, card) in enumerate(plays[:play_index]):
        if not 0 <= seat < 4:
            raise ValueError(f"第 {history_index + 1} 次出牌包含无效座位")
        if seat != state.turn:
            raise ValueError(
                f"第 {history_index + 1} 次出牌顺序错误："
                f"应为座位 {state.turn}，记录为 {seat}"
            )
        if card not in state.hands[seat]:
            raise ValueError(f"座位 {seat} 并不持有第 {history_index + 1} 次记录的牌")
        legal = rules.playable(state, state.hands[seat], seat)
        if card not in legal:
            raise ValueError(f"第 {history_index + 1} 次记录的是非法出牌")

        state.play_card_to_table(seat, card)
        if card.suit == Suit.SPADES:
            state.trump_broken = state.spades_broken = True
        state.turn = (seat + 1) % 4
        if state.trick_complete:
            winner = rules.winner_trick(state)
            state.complete_trick(winner)
            state.turn = state.trick_leader = winner

    target_seat, target_card = plays[play_index]
    if target_seat != state.turn:
        raise ValueError(
            f"目标动作应由座位 {state.turn} 执行，记录为座位 {target_seat}"
        )
    legal = rules.playable(state, state.hands[target_seat], target_seat)
    if target_card not in legal:
        raise ValueError("目标动作不是该局面的合法出牌")
    state.hand_bitsets = [cards_to_bitset(hand) for hand in state.hands]
    return state, (target_seat, target_card)


def analyze_replay_position(
    exact_solver: Any,
    initial_hands: Sequence[Sequence[Card]],
    max_bid: Sequence[Any],
    plays: Sequence[tuple[int, Card]],
    *,
    first_leader: int,
    play_index: int,
) -> dict[str, Any]:
    """Compute full-information root Q values for one replay action."""
    if exact_solver is None:
        raise RuntimeError("明手 solver 不可用")
    state, (target_seat, target_card) = build_replay_solver_state(
        initial_hands,
        max_bid,
        plays,
        first_leader=first_leader,
        play_index=play_index,
    )
    rules = SpadesRules()
    legal_cards = rules.playable(state, state.hands[target_seat], target_seat)

    started = time.perf_counter()
    raw_q = exact_solver.solve_with_q_fast(state)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    expanded_q = expand_equivalent_root_q_values(state, raw_q, legal_cards)
    missing = [card for card in legal_cards if card.card_id not in expanded_q]
    if missing:
        codes = ", ".join(card_to_code(card) for card in missing)
        raise RuntimeError(f"明手 solver 未返回所有合法动作的 Q 值：{codes}")
    if any(not math.isfinite(float(value)) for value in expanded_q.values()):
        raise RuntimeError("明手 solver 返回了非有限 Q 值")

    optimize_for_team = state.teams[target_seat]
    best_value = (
        max(expanded_q.values())
        if optimize_for_team == 0
        else min(expanded_q.values())
    )
    action_values = [
        {
            "card": card_to_code(card),
            "q": float(expanded_q[card.card_id]),
            "is_best": float(expanded_q[card.card_id]) == float(best_value),
            "is_played": card == target_card,
        }
        for card in legal_cards
    ]
    action_values.sort(
        key=lambda item: (item["q"], item["card"]),
        reverse=optimize_for_team == 0,
    )
    return {
        "schema_version": 1,
        "play_index": int(play_index),
        "trick_number": state.tricks_played + 1,
        "current_player": target_seat,
        "optimize_for_team": optimize_for_team,
        "played_card": card_to_code(target_card),
        "best_value": float(best_value),
        "action_q_values": action_values,
        "q_perspective": "team0_score_difference",
        "elapsed_ms": elapsed_ms,
    }


__all__ = [
    "REPLAY_ANALYSIS_FIRST_PLAY_INDEX",
    "REPLAY_ANALYSIS_LAST_PLAY_INDEX",
    "analyze_replay_position",
    "build_replay_solver_state",
    "card_to_code",
    "serialize_ai_play_info",
]
