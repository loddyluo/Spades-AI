"""
详细追踪精确求解器在SQ路径上的行为
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trace_exact2 import create_two_trick_state


# 第一步：模拟手动走SQ，然后追踪精确求解器在这个后续状态上的行为
state = create_two_trick_state()
rules = SpadesRules()
solver = ExactDoubleDummySolver()

# P0出SQ
action = Card.from_str("SQ")
new_state = solver._apply_action(state, action, 0)

print("P0出SQ后的状态:")
print(f"  轮到: 玩家{new_state.turn}")
for i in range(4):
    print(f"  玩家{i}手牌: {[str(c) for c in new_state.hands[i]]}")
print(f"  桌面: {[(p, str(c)) for p, c in new_state.table_cards]}")
print(f"  墩数: {new_state.tricks_played}")

# P1只能出SK (有黑桃必须跟)
print("\nP1必须跟黑桃，出SK后:")
new_state2 = solver._apply_action(new_state, Card.from_str("SK"), 1)
print(f"  轮到: 玩家{new_state2.turn}")
for i in range(4):
    print(f"  玩家{i}手牌: {[str(c) for c in new_state2.hands[i]]}")
print(f"  桌面: {[(p, str(c)) for p, c in new_state2.table_cards]}")

# P2可以选择任意牌（没有黑桃）
print("\nP2的合法动作:")
hand2 = new_state2.hands[2]
legal2 = rules.playable(new_state2, hand2, 2)
print(f"  {[str(c) for c in legal2]}")

for card2 in legal2:
    state_after_p2 = solver._apply_action(new_state2, card2, 2)
    print(f"\nP2出{card2}后:")
    print(f"  轮到: 玩家{state_after_p2.turn}")
    print(f"  桌面: {[(p, str(c)) for p, c in state_after_p2.table_cards]}")

    # P3可以选择任意牌（没有黑桃）
    hand3 = state_after_p2.hands[3]
    legal3 = rules.playable(state_after_p2, hand3, 3)

    for card3 in legal3:
        state_after_p3 = solver._apply_action(state_after_p2, card3, 3)
        print(f"    P3出{card3}: 轮到玩家{state_after_p3.turn}, 桌面={[(p, str(c)) for p, c in state_after_p3.table_cards]}")
        print(f"    tricks_won={state_after_p3.tricks_won}, 完成墩数={state_after_p3.tricks_played}")

        # 现在P1（赢家）领出
        # P1有C4, C5
        if state_after_p3.turn == 1:
            hand1 = state_after_p3.hands[1]
            legal1 = rules.playable(state_after_p3, hand1, 1)
            for card1 in legal1:
                state_after_p1 = solver._apply_action(state_after_p3, card1, 1)
                print(f"      P1出{card1}: 轮到{state_after_p1.turn}")

                # 继续追踪直到结束
                # ... 递归追踪比较复杂，直接调用精确求解器看最终结果
                final_value = solver.solve(state_after_p1)
                print(f"      精确求解器剩余价值 = {final_value}")
