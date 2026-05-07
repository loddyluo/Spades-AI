"""
15~30 张牌性能对比：稳定 C++ (`cpp_native`) vs 新尝试 (`cpp_opt1`)。

说明：
- 此文件用于“新优化尝试”的性能评估。
- 不影响主性能基准文件 tests/test_exact_solver_timing_15_35_cpp_native.py（仍只测主版本）。
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import time

sys.path.insert(0, '.')

from trick_taking.card import Suit
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummyPythonSolver
from trick_taking.solvers.exact_double_dummy_cpp_native import ExactDoubleDummyCppNativeSolver
from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver


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
    deck = Deck(STANDARD_52, seed=seed)
    rules = SpadesRules()
    hands = [deck.deal(13) for _ in range(4)]

    state = GameState()
    state.init_for_deal(4, hands, [], deck.all_cards)

    bids = []
    max_bid = []
    for pid in range(4):
        if rng.random() < 0.25:
            bid = 'nil' if rng.random() < 0.85 else 'blind_nil'
        else:
            bid = f'bid_{rng.randint(1, 13)}'
        bids.append(Bid(player_id=pid, value=bid))
        max_bid.append(bid)

    state.bids = bids
    state.max_bid = max_bid
    state.teams = [0, 1, 0, 1]
    state.phase = Phase.PLAYING
    state.turn = 0
    state.trick_leader = 0

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
    base = ExactDoubleDummyPythonSolver()
    native = ExactDoubleDummyCppNativeSolver()
    opt1 = ExactDoubleDummyCppOpt1Solver()
    if not base:
        raise RuntimeError("python 参考求解器不可用")
    if not native.native_available:
        raise RuntimeError("cpp_native 不可用")
    if not opt1.native_available:
        raise RuntimeError("cpp_opt1 不可用")

    print("=" * 120)
    print("15~30 张牌性能对比：cpp_native vs cpp_opt1（每个牌数3个样本）")
    print("=" * 120)

    matched_samples = 0
    tie_samples = 0

    for rem in range(15, 31):
        native_samples = []
        opt1_samples = []

        for i in range(3):
            seed = rem * 100 + i
            s1 = build_state_with_remaining_cards(rem, seed)
            s2 = build_state_with_remaining_cards(rem, seed)
            s_base = build_state_with_remaining_cards(rem, seed)

            b = base.solve_with_q(s_base)

            v1, t1 = timed_solve(native, s1)
            v2, t2 = timed_solve(opt1, s2)

            # 三方一致性检查：value / action_q_values 必须与 Python 参考对齐。
            if abs(b["value"] - v1) > 1e-9 or abs(b["value"] - v2) > 1e-9:
                raise AssertionError(
                    f"value mismatch remain={rem}, seed={seed}: python={b['value']}, native={v1}, opt1={v2}"
                )

            if set(b["action_q_values"].keys()) != set(native.solve_with_q(build_state_with_remaining_cards(rem, seed))["action_q_values"].keys()):
                raise AssertionError(f"action set mismatch remain={rem}, seed={seed}: python vs native")

            n_detail = native.solve_with_q(build_state_with_remaining_cards(rem, seed))
            o_detail = opt1.solve_with_q(build_state_with_remaining_cards(rem, seed))
            for action, q_value in b["action_q_values"].items():
                if abs(q_value - n_detail["action_q_values"][action]) > 1e-9:
                    raise AssertionError(
                        f"native q mismatch remain={rem}, seed={seed}, action={action}: python={q_value}, native={n_detail['action_q_values'][action]}"
                    )
                if abs(q_value - o_detail["action_q_values"][action]) > 1e-9:
                    raise AssertionError(
                        f"opt1 q mismatch remain={rem}, seed={seed}, action={action}: python={q_value}, opt1={o_detail['action_q_values'][action]}"
                    )

            best_q = max(b["action_q_values"].values()) if b["action_q_values"] else float("-inf")
            best_actions = [action for action, q_value in b["action_q_values"].items() if math.isclose(q_value, best_q, rel_tol=1e-9, abs_tol=1e-9)]
            if len(best_actions) > 1:
                tie_samples += 1

            native_samples.append(t1)
            opt1_samples.append(t2)
            matched_samples += 1

        if not native_samples or not opt1_samples:
            print(f"remain={rem:2d} | no comparable samples")
            continue

        navg = statistics.mean(native_samples)
        oavg = statistics.mean(opt1_samples)
        speedup = navg / oavg if oavg > 0 else float('inf')
        print(
            f"remain={rem:2d} | native_avg={navg:.6f}s | opt1_avg={oavg:.6f}s | speedup={speedup:.3f}x"
        )

    print("-" * 120)
    print(f"测试完成：已完成新优化尝试的 15~30 三方一致性与性能评估，匹配样本={matched_samples}，并列最优样本={tie_samples}。")


if __name__ == '__main__':
    main()
