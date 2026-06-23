#!/usr/bin/env python3
"""排查精确求解器为什么对 SJ/S7/S3 返回相同 Q 值。

方法: 手动执行每个动作, 进入下一状态, 对比求解器在新状态下的 value。
"""
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from trick_taking.game_state import GameState, TrickRecord, Phase
from trick_taking.card import Card, Suit, Rank
from trick_taking.games.spades import SpadesRules
from trick_taking.card import _STANDARD_CARDS as STANDARD_52
from trick_taking.solvers.exact_double_dummy_cpp_fastest import ExactDoubleDummyCppFastestSolver

# ── Sample #2 初始手牌 ──
sample_hands_str = [
    "DT H4 C5 HQ C7 D7 SA H5 SK S5 HT ST SQ",
    "DK HK C3 H7 CJ D3 CT D9 D2 S2 S9 C9 S4",
    "D4 HA CA H9 C2 DJ CQ H8 D5 S8 S6 H3 H6",
    "D8 S3 DQ S7 DA SJ C4 C6 H2 C8 CK HJ D6",
]

PLAY_SEQUENCE = [
    (0,"DT"),(1,"DK"),(2,"D4"),(3,"D6"),
    (1,"HK"),(2,"HA"),(3,"H2"),(0,"H4"),
    (2,"CA"),(3,"C4"),(0,"C5"),(1,"C3"),
    (2,"H9"),(3,"HJ"),(0,"HQ"),(1,"H7"),
    (0,"C7"),(1,"CJ"),(2,"C2"),(3,"C6"),
    (1,"D3"),(2,"DJ"),(3,"DA"),(0,"D7"),
    (3,"C8"),(0,"SA"),(1,"CT"),(2,"CQ"),
    (0,"H5"),(1,"D9"),(2,"H8"),(3,"CK"),
    (2,"D5"),(3,"DQ"),(0,"SK"),(1,"D2"),
    (0,"S5"),(1,"S2"),(2,"S8"),
]

def parse_hand(s):
    return [Card.from_str(c.strip()) for c in s.split()]

def build_state(hands, play_seq):
    state = GameState()
    state.init_for_deal(4, [list(h) for h in hands], [], list(STANDARD_52))
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    state.teams = [0, 1, 0, 1]
    state.max_bid = ["bid_5", "bid_1", "bid_2", "bid_3"]
    state.tricks_played = 9
    state.bids = []  # 避免 solver 检查 bids 字段

    rule = SpadesRules()
    tricks_raw = [play_seq[i:i+4] for i in range(0, 36, 4)]
    if len(play_seq) > 36:
        tricks_raw.append(play_seq[36:])

    for ti, trick_cards in enumerate(tricks_raw):
        if ti < 9:
            state.turn = trick_cards[0][0]
            state.trick_leader = trick_cards[0][0]
            state.table_cards = []
            for p, cs in trick_cards:
                card = Card.from_str(cs)
                if card.suit == Suit.SPADES:
                    state.spades_broken = True
                    state.trump_broken = True
                state.play_card_to_table(p, card)
            winner = rule.winner_trick(state)
            state.tricks_won[winner] += 1
            state.trick_history.append(TrickRecord(
                list(state.table_cards), winner, state.trick_leader,
            ))
            state.table_cards = []
        else:
            for p, cs in trick_cards:
                card = Card.from_str(cs)
                if card.suit == Suit.SPADES:
                    state.spades_broken = True
                    state.trump_broken = True
                state.play_card_to_table(p, card)
            state.turn = 3

    return state


def play_action_and_solve(state, action_card):
    """在 state 副本中执行 action_card, 完成当前墩, 然后在新状态上调用求解器。"""
    s = copy.deepcopy(state)
    rule = SpadesRules()

    # 执行动作
    s.play_card_to_table(s.turn, action_card)

    # 完成当前墩
    leader = s.trick_leader
    winner = rule.winner_trick(s)
    s.tricks_won[winner] += 1
    s.trick_history.append(TrickRecord(
        list(s.table_cards), winner, leader,
    ))
    s.table_cards = []
    s.turn = winner
    s.tricks_played += 1

    # 在新状态下求解
    solver = ExactDoubleDummyCppFastestSolver()
    result = solver.solve_with_q(s)
    return result


def main():
    hands = [parse_hand(s) for s in sample_hands_str]
    state = build_state(hands, PLAY_SEQUENCE)

    print("=== 当前桌面 ===")
    for p, c in state.table_cards:
        print(f"  P{p}: {c}")
    print(f"当前 turn: {state.turn}")
    print(f"剩余牌: P0={[str(c) for c in state.hands[0]]}, "
          f"P1={[str(c) for c in state.hands[1]]}, "
          f"P2={[str(c) for c in state.hands[2]]}, "
          f"P3={[str(c) for c in state.hands[3]]}")

    # ── 测试 1: 直接在当前状态调用 solve_with_q ──
    solver = ExactDoubleDummyCppFastestSolver()
    result_before = solver.solve_with_q(state)
    print(f"\n=== 当前状态 solve_with_q ===")
    print(f"  value={result_before['value']}, optimize_for_team={result_before['optimize_for_team']}")
    for card, q in result_before["action_q_values"].items():
        print(f"  {card}: q={q}")

    # ── 测试 2: 手动执行 SJ, S7, S3 看下一状态 value ──
    print(f"\n=== 执行每个动作后的下一状态 ===")
    for action_str in ["S7", "S3", "SJ"]:
        card = Card.from_str(action_str)
        if card not in state.hands[state.turn]:
            print(f"  {action_str}: 不在手牌中, 跳过")
            continue
        next_result = play_action_and_solve(state, card)
        print(f"\n  执行 {action_str} 后:")
        print(f"    value={next_result['value']}, "
              f"current_player={next_result['current_player']}, "
              f"optimize_for_team={next_result['optimize_for_team']}")
        if next_result.get("action_q_values"):
            print(f"    下一状态的动作 Q 值:")
            for c, q in sorted(next_result["action_q_values"].items(),
                              key=lambda x: -x[1])[:5]:
                print(f"      {c}: q={q}")

    # ── 测试 3: 手动深搜两步, 验证最优路线 ──
    print(f"\n=== 两步深搜: 先出 X, 对方最优应对, 再看最终 value ===")
    for action_str in ["S7", "S3", "SJ"]:
        card = Card.from_str(action_str)
        if card not in state.hands[state.turn]:
            continue

        # 第一步: P3 出 action
        s1 = copy.deepcopy(state)
        s1.play_card_to_table(s1.turn, card)

        # 完成墩10, 计算赢家
        leader = s1.trick_leader
        winner1 = SpadesRules().winner_trick(s1)
        s1.tricks_won[winner1] += 1
        s1.trick_history.append(TrickRecord(
            list(s1.table_cards), winner1, leader,
        ))
        s1.table_cards = []
        s1.turn = winner1
        s1.tricks_played += 1

        print(f"\n  出 {action_str} → 墩10 赢家: P{winner1}")

        # 在新状态上求解（P_winner1 引牌）
        r1 = solver.solve_with_q(s1)
        print(f"    下一状态 value={r1['value']} (team {r1['optimize_for_team']})")
        print(f"    P{winner1} 合法动作的 Q 值:")
        for c, q in sorted(r1["action_q_values"].items(), key=lambda x: -x[1]):
            print(f"      {c}: q={q}")

        # 第二步: 对方选最优动作
        opt_team = r1['optimize_for_team']
        if r1["action_q_values"]:
            if opt_team == 0:
                best_next = max(r1["action_q_values"], key=r1["action_q_values"].get)
            else:
                best_next = min(r1["action_q_values"], key=r1["action_q_values"].get)
            print(f"    → P{winner1} 最优动作: {best_next} (q={r1['action_q_values'][best_next]})")

            # 执行这个最优动作
            s2 = copy.deepcopy(s1)
            s2.play_card_to_table(s2.turn, best_next)
            # 不完成墩, 直接看新状态
            r2 = solver.solve_with_q(s2)
            print(f"    执行 {best_next} 后 value={r2['value']}")

    # ── 测试 4: 直接对比 solve() 的返回值 ──
    print(f"\n=== 对比 solve() (纯 value, 无 Q) ===")
    for action_str in ["S7", "S3", "SJ"]:
        card = Card.from_str(action_str)
        if card not in state.hands[state.turn]:
            continue
        s_copy = copy.deepcopy(state)
        s_copy.play_card_to_table(s_copy.turn, card)
        winner = SpadesRules().winner_trick(s_copy)
        s_copy.tricks_won[winner] += 1
        s_copy.trick_history.append(TrickRecord(
            list(s_copy.table_cards), winner, s_copy.trick_leader,
        ))
        s_copy.table_cards = []
        s_copy.turn = winner
        s_copy.tricks_played += 1
        value = solver.solve(s_copy)
        print(f"  执行 {action_str} → 赢家 P{winner}, 后续 state value = {value}")


if __name__ == "__main__":
    main()
