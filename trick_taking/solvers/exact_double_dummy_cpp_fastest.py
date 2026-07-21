"""
极速 C++ 精确双明手求解器包装器。

优化总览：
- Zobrist Hashing (增量更新)
- State Normalization (Rank Canonicalization, TT命中率暴增)
- Quick Tricks Pruning (确定赢墩提前剪枝)
- TT Only at Trick Boundaries (减少缓存污染)
- Zero Heap Allocation (全栈上固定数组)
- Position-Aware Move Ordering (领牌/跟牌分别排序)
- Equivalent Card Filtering
- PVS (Principal Variation Search)
- Killer Move + History Heuristic
- MTD(f) (根节点零窗口迭代)
- Multi-threading (根节点并行)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from typing import Any, Dict

from trick_taking.card import Card, Rank, Suit
from trick_taking.game_state import GameState
from trick_taking.solvers._native_compile import (
    FASTEST_BUILD_RECIPE,
    compile_fastest_solver,
)
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.solvers.native_lib_loader import (
    NATIVE_LIBRARY_ABI_VERSION,
    ensure_native_library,
)


# The fastest C++ library keeps process-wide mutable transposition tables and
# generation counters.  CDLL calls release the GIL, so every wrapper instance
# in this process must share one lock around native entry points.
_NATIVE_SOLVER_LOCK = threading.RLock()
_FASTEST_REQUIRED_SYMBOLS = (
    "solve_native",
    "solve_native_with_q",
    "analyze_forced_outcome_native",
)


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


class _ForcedOutcomeResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("team0_final_tricks", ctypes.c_int32),
        ("nil_broken_mask", ctypes.c_uint32),
        ("nodes_searched", ctypes.c_uint64),
        ("elapsed_ms", ctypes.c_double),
    ]


class ExactDoubleDummyCppFastestSolver(ExactDoubleDummySolver):
    """极速 C++ 精确双明手求解器。"""

    def __init__(self):
        super().__init__()
        self._lib = None
        self._ensure_library()

    def _ensure_library(self) -> None:
        this_dir = os.path.dirname(__file__)

        try:
            lib_path = ensure_native_library(
                this_dir,
                "_exact_double_dummy_cpp_fastest_core",
                "exact_double_dummy_cpp_fastest_core.cpp",
                compile_fastest_solver,
                required_symbols=_FASTEST_REQUIRED_SYMBOLS,
                abi_version=NATIVE_LIBRARY_ABI_VERSION,
                build_recipe=FASTEST_BUILD_RECIPE,
            )

            self._lib = ctypes.CDLL(lib_path)
            self._lib.solve_native.argtypes = [ctypes.POINTER(_NativeState)]
            self._lib.solve_native.restype = ctypes.c_double
            self._lib.solve_native_with_q.argtypes = [
                ctypes.POINTER(_NativeState),
                ctypes.POINTER(_RootQResult),
            ]
            self._lib.solve_native_with_q.restype = None
            self._lib.analyze_forced_outcome_native.argtypes = [
                ctypes.POINTER(_NativeState),
                ctypes.c_int64,
                ctypes.POINTER(_ForcedOutcomeResult),
            ]
            self._lib.analyze_forced_outcome_native.restype = None
        except Exception as e:
            print(f"Warning: Failed to build/load fastest solver: {e}", file=sys.stderr)
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
            raise RuntimeError("极速 C++ 求解器不可用")

        native_state = self._to_native_state(state)
        with _NATIVE_SOLVER_LOCK:
            return float(self._lib.solve_native(ctypes.byref(native_state)))

    def solve_with_q(self, state: GameState) -> Dict[str, Any]:
        self._validate_state(state)
        if not self.native_available:
            raise RuntimeError("极速 C++ 求解器不可用")

        native_state = self._to_native_state(state)
        out = _RootQResult()
        with _NATIVE_SOLVER_LOCK:
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

    def solve_with_q_fast(self, state: GameState) -> Dict[int, float]:
        """返回 {card_id: q_value} 的简化格式，供 rule_based 并行 solver 使用。"""
        self._validate_state(state)
        if not self.native_available:
            raise RuntimeError("极速 C++ 求解器不可用")

        native_state = self._to_native_state(state)
        out = _RootQResult()
        with _NATIVE_SOLVER_LOCK:
            self._lib.solve_native_with_q(ctypes.byref(native_state), ctypes.byref(out))

        result: Dict[int, float] = {}
        for idx in range(int(out.count)):
            card_id = int(out.actions[idx])
            result[card_id] = float(out.q_values[idx])
        return result

    @staticmethod
    def _forced_timeout_result(started: float) -> Dict[str, object]:
        return {
            "status": "timeout",
            "team0_final_tricks": -1,
            "nil_broken_mask": 0,
            "nodes_searched": 0,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
        }

    def analyze_forced_outcome(
        self,
        state: GameState,
        time_budget_seconds: float = 1.0,
    ) -> Dict[str, object]:
        """Prove whether every legal continuation has one terminal signature.

        The wall-clock budget includes waiting for the native solver lock.  A
        timeout is inconclusive and is never interpreted as permission to
        reveal hidden cards.
        """
        self._validate_state(state)
        started = time.monotonic()
        budget = max(0.0, float(time_budget_seconds))
        if budget == 0.0 or not self.native_available:
            return self._forced_timeout_result(started)

        acquired = _NATIVE_SOLVER_LOCK.acquire(timeout=budget)
        if not acquired:
            return self._forced_timeout_result(started)

        try:
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0.0:
                return self._forced_timeout_result(started)

            native_state = self._to_native_state(state)
            out = _ForcedOutcomeResult()
            self._lib.analyze_forced_outcome_native(
                ctypes.byref(native_state),
                max(1, int(remaining * 1_000_000)),
                ctypes.byref(out),
            )
        finally:
            _NATIVE_SOLVER_LOCK.release()

        labels = {1: "fixed", 2: "variable", 3: "timeout"}
        status = labels.get(int(out.status), "timeout")
        return {
            "status": status,
            "team0_final_tricks": int(out.team0_final_tricks),
            "nil_broken_mask": int(out.nil_broken_mask),
            "nodes_searched": int(out.nodes_searched),
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
        }
