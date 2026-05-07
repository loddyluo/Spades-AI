"""
原生 C++ 稳定版本耗时测试（剩余15~35张，每个牌数3个样本）。

说明：
- 仅测试速度最快且稳定的 C++ 求解器 `ExactDoubleDummyCppNativeSolver`。
- 不对比 Python 或其他尝试版本，节省时间。
"""

from __future__ import annotations

import random
import statistics
import sys
import time

sys.path.insert(0, '.')

from trick_taking.card import Suit
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_native import ExactDoubleDummyCppNativeSolver
from trick_taking.utils.state_tools import create_random_state


def apply_action(state, action, player_id, rules):
    state.play_card_to_table(player_id, action)
    if action.suit == Suit.SPADES:
        state.spades_broken = True
        state.trump_broken = True
    state.turn = (player_id + 1) % state.num_players
    if state.trick_complete:
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.trick_leader = winner
        state.turn = winner


def build_state_with_remaining_cards(target_remaining: int, seed: int):
    rng = random.Random(seed)
    rules = SpadesRules()
    saved_state = random.getstate()
    random.seed(seed)
    state = create_random_state()
    random.setstate(saved_state)

    while sum(len(h) for h in state.hands) > target_remaining:
        pid = state.turn
        legal = rules.playable(state, state.hands[pid], pid)
        if not legal:
            break
        apply_action(state, rng.choice(legal), pid, rules)

    return state


def timed_solve(solver, state):
    start = time.perf_counter()
    value = solver.solve(state)
    return value, time.perf_counter() - start


def main():
    solver = ExactDoubleDummyCppNativeSolver()
    if not solver.native_available:
        raise RuntimeError("C++ 原生实现不可用")

    print("=" * 120)
    print("原生 C++ 稳定版本耗时测试（剩余15~35张，每个牌数3个样本）")
    print("=" * 120)

    for rem in range(15, 31):
        samples = []
        for i in range(3):
            seed = rem * 100 + i
            state = build_state_with_remaining_cards(rem, seed)
            actual = sum(len(h) for h in state.hands)
            if actual != rem:
                raise AssertionError(f"状态构造失败: target={rem}, actual={actual}, seed={seed}")
            value, elapsed = timed_solve(solver, state)
            samples.append(elapsed)
            _ = value

        avg = statistics.mean(samples)
        print(f"remain={rem:2d} | avg={avg:.6f}s | min={min(samples):.6f}s | max={max(samples):.6f}s")

    print("-" * 120)
    print("测试完成：仅统计稳定 C++ 版本。")


if __name__ == '__main__':
    main()
