"""
排查精确求解器的潜在bug：每次创建新求解器来避免缓存污染
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trace_exact2 import create_two_trick_state


state = create_two_trick_state()
rules = SpadesRules()

# 方法1：使用同一个求解器（可能走缓存）
print("=" * 60)
print("方法1：使用同一个求解器（缓存共享）")
solver1 = ExactDoubleDummySolver()
root_value = solver1.solve(state)
print(f"根节点最优值: {root_value}")

hand0 = state.hands[0]
for action in rules.playable(state, hand0, 0):
    new_state = solver1._apply_action(state, action, 0)
    value = solver1.solve(new_state)
    print(f"  出 {action}: 后续最优值 = {value}")

# 方法2：每个子节点用全新求解器（无缓存）
print("\n" + "=" * 60)
print("方法2：每个子节点用全新求解器（无缓存影响）")
for action in rules.playable(state, hand0, 0):
    new_state = solver1._apply_action(state, action, 0)
    fresh_solver = ExactDoubleDummySolver()
    value = fresh_solver.solve(new_state)
    print(f"  出 {action}: 后续最优值 = {value}")

# 方法3：完全不用缓存
print("\n" + "=" * 60)
print("方法3：不用缓存，直接模拟SQ路线到终局")

from copy import deepcopy

def simulate_sq_line():
    s = create_two_trick_state()
    # 墩1: SQ, SK, P2出H2, P3出H9 -> P1赢
    s.play_card_to_table(0, Card.from_str("SQ"))
    s.play_card_to_table(1, Card.from_str("SK"))
    s.play_card_to_table(2, Card.from_str("H2"))
    s.play_card_to_table(3, Card.from_str("H9"))
    # 计算赢家
    spades = [(p,c) for p,c in s.table_cards if c.suit == Card.from_str("SK").suit]
    winner, _ = max(spades, key=lambda x: x[1].rank.value)
    print(f"墩1赢家: P{winner}")
    s.complete_trick(winner)
    s.trick_leader = winner
    s.turn = winner
    print(f"tricks_won={s.tricks_won}, tricks_played={s.tricks_played}")
    print(f"P1手牌: {[str(c) for c in s.hands[1]]}")

    # 墩2: P1出C4, P2出..., P3出..., P0出...
    s.play_card_to_table(1, Card.from_str("C4"))
    s.play_card_to_table(2, Card.from_str("HT"))
    s.play_card_to_table(3, Card.from_str("H8"))
    s.play_card_to_table(0, Card.from_str("H3"))
    clubs = [(p,c) for p,c in s.table_cards if c.suit == Card.from_str("C4").suit]
    winner, _ = max(clubs, key=lambda x: x[1].rank.value)
    print(f"墩2赢家: P{winner}")
    s.complete_trick(winner)
    s.trick_leader = winner
    s.turn = winner
    print(f"tricks_won={s.tricks_won}, tricks_played={s.tricks_played}")
    print(f"P1手牌: {[str(c) for c in s.hands[1]]}")

    # 墩3: P1出C5, P2出..., P3出..., P0出...
    s.play_card_to_table(1, Card.from_str("C5"))
    s.play_card_to_table(2, Card.from_str("HJ"))
    s.play_card_to_table(3, Card.from_str("H9"))  # already played H9 in trick1?
    # Wait, we already played H9 in trick 1! Error!
    print(f"错误：H9已经在墩1打了！")

# Wait, I need to track what's already been played. Let me be more careful.
print("\n" + "=" * 60)
print("方法4：用精确模拟验证SQ路线")

def simulate_carefully():
    s = create_two_trick_state()
    # 初始手牌
    print("初始手牌:")
    for i in range(4):
        print(f"  P{i}: {[str(c) for c in s.hands[i]]}")

    # 墩1: P0出SQ, P1出SK(必须跟), P2出H2, P3出H6 -> P1赢(SK>SQ)
    plays1 = [(0, "SQ"), (1, "SK"), (2, "H2"), (3, "H6")]
    for pid, c in plays1:
        card = Card.from_str(c)
        s.play_card_to_table(pid, card)
        print(f"P{pid}出{c}: 桌面{[(p,str(cc)) for p,cc in s.table_cards]}")
    # 确定赢家
    sp = [(p,cc) for p,cc in s.table_cards if cc.suit == Card.from_str("SK").suit]
    w1, _ = max(sp, key=lambda x: x[1].rank.value)
    print(f"墩1赢家: P{w1} (SK > SQ)")
    s.complete_trick(w1)
    s.trick_leader = w1
    s.turn = w1

    print(f"\n墩1后:")
    for i in range(4):
        print(f"  P{i}: {[str(c) for c in s.hands[i]]}")
    print(f"tricks_won={s.tricks_won}")

    # 墩2: P1出C4, P2出HT, P3出H8, P0出H3 -> P1赢(唯一梅花)
    plays2 = [(1, "C4"), (2, "HT"), (3, "H8"), (0, "H3")]
    for pid, c in plays2:
        card = Card.from_str(c)
        s.play_card_to_table(pid, card)
        print(f"P{pid}出{c}: 桌面{[(p,str(cc)) for p,cc in s.table_cards]}")
    clubs = [(p,cc) for p,cc in s.table_cards if cc.suit == Card.from_str("C4").suit]
    w2, _ = max(clubs, key=lambda x: x[1].rank.value)
    print(f"墩2赢家: P{w2}")
    s.complete_trick(w2)
    s.trick_leader = w2
    s.turn = w2

    print(f"\n墩2后:")
    for i in range(4):
        print(f"  P{i}: {[str(c) for c in s.hands[i]]}")
    print(f"tricks_won={s.tricks_won}")

    # 墩3: P1出C5, P2出HJ, P3出H9, P0出SA -> P1赢(唯一梅花)
    # P0还剩SA和H3? 不对, H3已经在墩2打了. P0还剩SA
    # P3还剩H9
    # P2还剩HJ
    plays3 = [(1, "C5"), (2, "HJ"), (3, "H9"), (0, "SA")]
    for pid, c in plays3:
        card = Card.from_str(c)
        s.play_card_to_table(pid, card)
        print(f"P{pid}出{c}: 桌面{[(p,str(cc)) for p,cc in s.table_cards]}")
    clubs = [(p,cc) for p,cc in s.table_cards if cc.suit == Card.from_str("C5").suit]
    w3, _ = max(clubs, key=lambda x: x[1].rank.value)
    print(f"墩3赢家: P{w3}")
    s.complete_trick(w3)
    s.trick_leader = w3
    s.turn = w3

    print(f"\n终局:")
    print(f"tricks_won={s.tricks_won}")
    print(f"tricks_played={s.tricks_played}")

    scores = rules.score(s)
    print(f"scores={scores}")
    print(f"P0得分差 = {scores[0]}")

simulate_carefully()
