"""
原生 C++ 精确双明手求解器包装器。

说明：
- 该版本把搜索核心放到 C++ 中执行，Python 仅负责状态转换与接口包装。
- 当前先实现 solve / solve_with_q：
  - solve: 直接调用 C++ 原生搜索得到最优值。
  - solve_with_q: 在根节点枚举动作，逐个调用 solve 生成 action_q_values。

注意：
- 该实现独立于基线接口，不覆盖现有 solver。
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Dict

from trick_taking.card import Card, Rank, Suit
from trick_taking.game_state import GameState
from trick_taking.solvers._native_compile import (
    NATIVE_BUILD_RECIPE,
    compile_native_solver,
)
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.solvers.native_lib_loader import (
    NATIVE_LIBRARY_ABI_VERSION,
    ensure_native_library,
)


_NATIVE_REQUIRED_SYMBOLS = ("solve_native", "solve_native_with_q")


class _NativeState(ctypes.Structure):
    _fields_ = [
        ("num_players", ctypes.c_int32),
        ("hand_bits", ctypes.c_uint64 * 4),
        ("hand_counts", ctypes.c_int32 * 4),
        ("table_pids", ctypes.c_int32 * 4),
        ("table_suits", ctypes.c_int32 * 4),
        ("table_ranks", ctypes.c_int32 * 4),
        ("table_count", ctypes.c_int32),
        ("turn", ctypes.c_int32),
        ("trick_leader", ctypes.c_int32),
        ("spades_broken", ctypes.c_int32),
        ("tricks_played", ctypes.c_int32),
        ("tricks_won", ctypes.c_int32 * 4),
        ("max_bid", ctypes.c_int32 * 4),
        ("teams", ctypes.c_int32 * 4),
    ]


class _RootQResult(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int32),
        ("current_player", ctypes.c_int32),
        ("optimize_for_team", ctypes.c_int32),
        ("best_action", ctypes.c_int32),
        ("value", ctypes.c_double),
        ("actions", ctypes.c_int32 * 13),
        ("q_values", ctypes.c_double * 13),
    ]


class ExactDoubleDummyCppNativeSolver(ExactDoubleDummySolver):
    """原生 C++ 搜索内核尝试版。"""

    def __init__(self):
        super().__init__()
        self._lib = None
        self._ensure_library()

    def _ensure_library(self) -> None:
        this_dir = os.path.dirname(__file__)

        try:
            lib_path = ensure_native_library(
                this_dir,
                "_exact_double_dummy_cpp_native_core",
                "exact_double_dummy_cpp_native_core.cpp",
                compile_native_solver,
                required_symbols=_NATIVE_REQUIRED_SYMBOLS,
                abi_version=NATIVE_LIBRARY_ABI_VERSION,
                build_recipe=NATIVE_BUILD_RECIPE,
            )

            self._lib = ctypes.CDLL(lib_path)
            self._lib.solve_native.argtypes = [ctypes.POINTER(_NativeState)]
            self._lib.solve_native.restype = ctypes.c_double
            self._lib.solve_native_with_q.argtypes = [
                ctypes.POINTER(_NativeState),
                ctypes.POINTER(_RootQResult),
            ]
            self._lib.solve_native_with_q.restype = None
        except Exception:
            self._lib = None

    @property
    def native_available(self) -> bool:
        return self._lib is not None

    def _to_native_state(self, state: GameState) -> _NativeState:
        ns = _NativeState()
        ns.num_players = state.num_players
        for i in range(4):
            ns.hand_bits[i] = int(state.hand_bitsets[i]) if i < len(state.hand_bitsets) else 0
            ns.hand_counts[i] = len(state.hands[i]) if i < len(state.hands) else 0
            ns.tricks_won[i] = int(state.tricks_won[i]) if i < len(state.tricks_won) else 0
            ns.max_bid[i] = self._bid_to_native(state.max_bid[i] if i < len(state.max_bid) else None)
            ns.teams[i] = int(state.teams[i]) if i < len(state.teams) else 0

        ns.table_count = len(state.table_cards)
        for idx in range(4):
            if idx < ns.table_count:
                pid, card = state.table_cards[idx]
                ns.table_pids[idx] = pid
                ns.table_suits[idx] = card.suit.value
                ns.table_ranks[idx] = card.rank.value
            else:
                ns.table_pids[idx] = 0
                ns.table_suits[idx] = 0
                ns.table_ranks[idx] = 2

        ns.turn = state.turn
        ns.trick_leader = state.trick_leader
        ns.spades_broken = 1 if state.spades_broken else 0
        ns.tricks_played = state.tricks_played
        return ns

    @staticmethod
    def _bid_to_native(value: Any) -> int:
        if value is None:
            return 0
        if value == "nil":
            return 0
        if value == "blind_nil":
            return 14
        if isinstance(value, str) and value.startswith("bid_"):
            return int(value.split("_")[1])
        if isinstance(value, int):
            return value
        return 0

    @staticmethod
    def _card_from_id(card_id: int) -> Card:
        return Card(Suit(card_id // 13), Rank((card_id % 13) + 2))

    def solve(self, state: GameState) -> float:
        self._validate_state(state)
        if not self.native_available:
            raise RuntimeError("原生 C++ 求解器不可用，无法回退到 Python 基线")

        native_state = self._to_native_state(state)
        return float(self._lib.solve_native(ctypes.byref(native_state)))

    def solve_with_q(self, state: GameState) -> Dict[str, Any]:
        self._validate_state(state)
        if not self.native_available:
            raise RuntimeError("原生 C++ 求解器不可用，无法回退到 Python 基线")

        native_state = self._to_native_state(state)
        out = _RootQResult()
        self._lib.solve_native_with_q(ctypes.byref(native_state), ctypes.byref(out))

        action_q_values: Dict[Card, float] = {}
        action_values = []
        for idx in range(int(out.count)):
            action = self._card_from_id(int(out.actions[idx]))
            q_value = float(out.q_values[idx])
            action_q_values[action] = q_value
            action_values.append({"action": action, "q_value": q_value})

        optimize_for_team = int(out.optimize_for_team)
        action_values.sort(key=lambda x: x["q_value"], reverse=(optimize_for_team == 0))

        best_action = None
        if int(out.best_action) >= 0:
            best_action = self._card_from_id(int(out.best_action))

        return {
            "value": float(out.value),
            "best_action": best_action,
            "action_q_values": action_q_values,
            "action_values": action_values,
            "current_player": int(out.current_player),
            "optimize_for_team": optimize_for_team,
        }
