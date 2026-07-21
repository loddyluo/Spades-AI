from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from trick_taking.card import Card, Rank, Suit, cards_to_bitset
from trick_taking.game_state import GameState, Phase
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


class _OverlapDetectingNativeLibrary:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.entered = threading.Event()
        self.active = 0
        self.max_active = 0
        self.forced_calls = 0

    def solve_native(self, native_state_pointer) -> float:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            time.sleep(0.05)
            return 0.0
        finally:
            with self._guard:
                self.active -= 1

    def analyze_forced_outcome_native(
        self, native_state_pointer, budget_microseconds, output_pointer
    ) -> None:
        self.forced_calls += 1


def _solver_state() -> GameState:
    cards = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.HEARTS, Rank.FOUR),
        Card(Suit.HEARTS, Rank.FIVE),
    ]
    state = GameState()
    state.num_players = 4
    state.phase = Phase.PLAYING
    state.hands = [[card] for card in cards]
    state.hand_bitsets = [cards_to_bitset(hand) for hand in state.hands]
    state.max_bid = ["bid_1", "bid_1", "bid_1", "bid_1"]
    state.teams = [0, 1, 0, 1]
    state.table_cards = []
    state.turn = 0
    state.trick_leader = 0
    state.spades_broken = False
    state.tricks_played = 12
    state.tricks_won = [3, 3, 3, 3]
    return state


def test_native_solver_calls_are_serialized_across_wrapper_instances() -> None:
    native_library = _OverlapDetectingNativeLibrary()
    first = ExactDoubleDummyCppFastestSolver.__new__(ExactDoubleDummyCppFastestSolver)
    second = ExactDoubleDummyCppFastestSolver.__new__(ExactDoubleDummyCppFastestSolver)
    first._lib = native_library
    second._lib = native_library
    state = _solver_state()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda solver: solver.solve(state), [first, second]))

    assert results == [0.0, 0.0]
    assert native_library.max_active == 1


def test_forced_outcome_budget_includes_waiting_for_native_lock() -> None:
    native_library = _OverlapDetectingNativeLibrary()
    blocker = ExactDoubleDummyCppFastestSolver.__new__(ExactDoubleDummyCppFastestSolver)
    checker = ExactDoubleDummyCppFastestSolver.__new__(ExactDoubleDummyCppFastestSolver)
    blocker._lib = native_library
    checker._lib = native_library
    state = _solver_state()

    with ThreadPoolExecutor(max_workers=1) as executor:
        blocking_call = executor.submit(blocker.solve, state)
        assert native_library.entered.wait(timeout=1.0)
        result = checker.analyze_forced_outcome(state, time_budget_seconds=0.01)
        assert blocking_call.result() == 0.0

    assert result["status"] == "timeout"
    assert result["elapsed_ms"] < 100.0
    assert native_library.forced_calls == 0
