#!/usr/bin/env python3
"""Correctness and benchmark test for the C++ exact solver.

Generates random game states at various remaining card counts, runs both
the Python reference solver and the C++ solver, and compares results.
Then benchmarks the C++ solver at each level.
"""
import sys, time, random
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState, Phase, Bid
from trick_taking.games.spades import SpadesRules
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummyPythonSolver
from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver


def build_state(remaining, seed=42):
    rng = random.Random(seed)
    deck = Deck(STANDARD_52, seed=seed)
    hands = [deck.deal(13) for _ in range(4)]
    state = GameState()
    state.init_for_deal(4, hands, [], deck.all_cards)
    state.teams = [0, 1, 0, 1]
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    state.turn = 0
    state.trick_leader = 0
    state.max_bid = ['bid_3', 'bid_3', 'bid_3', 'bid_3']
    state.bids = [Bid(player_id=i, value='bid_3') for i in range(4)]
    rules = SpadesRules()
    to_play = 52 - remaining
    played = 0
    while played < to_play and not rules.end_trickgame(state):
        cur = state.turn
        legal = rules.playable(state, state.hands[cur], cur)
        if not legal:
            break
        card = rng.choice(legal)
        state.play_card_to_table(cur, card)
        if card.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True
        state.turn = (cur + 1) % 4
        if state.trick_complete:
            winner = rules.winner_trick(state)
            state.complete_trick(winner)
            state.trick_leader = winner
            state.turn = winner
        played += 1
    return state


def main():
    cpp = ExactDoubleDummyCppOpt1Solver()
    assert cpp.native_available, "C++ solver not available!"
    py = ExactDoubleDummyPythonSolver()

    # === Correctness ===
    print("=== Correctness Check (C++ vs Python reference) ===")
    errors = 0
    for remaining in [4, 8, 12, 16, 20, 24]:
        for seed in range(5):
            state = build_state(remaining, seed=seed * 137 + 7)
            actual = sum(len(h) for h in state.hands)
            if abs(actual - remaining) > 2:
                continue

            py.tt.clear()
            v_py = py.solve(state)
            v_cpp = cpp.solve(state)

            if abs(v_py - v_cpp) > 0.01:
                print("  MISMATCH rem=%d seed=%d: py=%.1f cpp=%.1f" % (remaining, seed, v_py, v_cpp))
                errors += 1
            else:
                pass  # OK

            # Also check solve_with_q best_action matches
            py.tt.clear()
            q_py = py.solve_with_q(state)
            q_cpp = cpp.solve_with_q(state)
            if q_py['best_action'] is not None and q_cpp['best_action'] is not None:
                if q_py['best_action'].card_id != q_cpp['best_action'].card_id:
                    # Check if values are the same (could be tie)
                    py_best_v = q_py['action_q_values'].get(q_py['best_action'], None)
                    cpp_best_v = q_cpp['action_q_values'].get(q_cpp['best_action'], None)
                    if py_best_v is not None and cpp_best_v is not None and abs(py_best_v - cpp_best_v) > 0.01:
                        print("  Q-MISMATCH rem=%d seed=%d: py_best=%s(%.1f) cpp_best=%s(%.1f)" % (
                            remaining, seed, q_py['best_action'], py_best_v,
                            q_cpp['best_action'], cpp_best_v))
                        errors += 1

    if errors == 0:
        print("  All checks passed!")
    else:
        print("  %d ERRORS found!" % errors)

    # === Benchmark ===
    print("\n=== Benchmark (C++ solver) ===")
    print("%5s %12s %12s %12s %12s %6s" % ("Rem", "solve avg", "solve max", "w_q avg", "w_q max", "Acts"))
    print("-" * 65)

    for remaining in [4, 8, 12, 16, 20, 24, 28]:
        t_solve = []
        t_q = []
        acts = 0
        for seed in range(5):
            state = build_state(remaining, seed=seed * 137)
            actual = sum(len(h) for h in state.hands)
            if abs(actual - remaining) > 2:
                continue

            t0 = time.perf_counter()
            cpp.solve(state)
            t1 = time.perf_counter()
            t_solve.append(t1 - t0)

            if remaining <= 28:
                t0 = time.perf_counter()
                r = cpp.solve_with_q(state)
                t1 = time.perf_counter()
                t_q.append(t1 - t0)
                acts = len(r.get('action_q_values', {}))

        if t_solve:
            avg_s = sum(t_solve) / len(t_solve) * 1000
            max_s = max(t_solve) * 1000
            if t_q:
                avg_q = sum(t_q) / len(t_q) * 1000
                max_q = max(t_q) * 1000
            else:
                avg_q = max_q = float('nan')
            print("%5d %9.1fms %9.1fms %9.1fms %9.1fms %6d" % (
                remaining, avg_s, max_s, avg_q, max_q, acts))
        sys.stdout.flush()

    # single solve() for 32
    print("\nremaining=32 (1 sample, solve only):")
    state = build_state(32, seed=0)
    t0 = time.perf_counter()
    cpp.solve(state)
    t1 = time.perf_counter()
    print("  solve(): %.2fs" % (t1 - t0))


if __name__ == "__main__":
    main()
