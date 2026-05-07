"""
调试rollout在SQ路线上的行为
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.double_dummy import DoubleDummySolver
from trace_exact2 import create_two_trick_state

state = create_two_trick_state()
rules = SpadesRules()
solver = DoubleDummySolver(max_iterations=1)  # 只用来访问方法

# 直接测试rollout从SQ状态开始的表现
# 第一步：出SQ
sq_state = solver._apply_action(state, Card.from_str("SQ"), 0)

print("测试SQ出牌后的rollout:")
print(f"P0手牌: {[str(c) for c in sq_state.hands[0]]}")
print(f"P1手牌: {[str(c) for c in sq_state.hands[1]]}")
print(f"P2手牌: {[str(c) for c in sq_state.hands[2]]}")
print(f"P3手牌: {[str(c) for c in sq_state.hands[3]]}")

# 模拟整个rollout过程
def trace_rollout(solver, state, root_player):
    """追踪模拟过程"""
    sim = solver._deep_copy_state(state)
    step = 0
    max_steps = 20
    print(f"\n开始模拟（root_player={root_player}）:")

    while not rules.end_trickgame(sim) and step < max_steps:
        step += 1
        cp = sim.turn
        print(f"\n步骤{step}: 玩家{cp}（队伍{sim.teams[cp]}，叫{sim.max_bid[cp]}）")
        legal = rules.playable(sim, sim.hands[cp], cp)
        print(f"  手牌: {[str(c) for c in sim.hands[cp]]}")
        print(f"  合法动作: {[str(c) for c in legal]}")

        # 显示动作评估
        for act in legal:
            next_s = solver._apply_action(sim, act, cp)
            val = solver._rollout_state_value(next_s, root_player)

            # 检查是否为浪费将牌
            is_waste = (act.suit == Suit.SPADES
                       and sim.table_cards
                       and sim.table_cards[0][1].suit != Suit.SPADES)
            is_nil = sim.max_bid[cp] in ('nil', 'blind_nil')
            root_team = sim.teams[root_player]
            maximize = sim.teams[cp] == root_team

            penalty_str = ""
            if is_waste and maximize and not is_nil:
                pen = act.rank.value
                val -= pen
                penalty_str = f"（将牌惩罚-{pen}）"

            is_nil_str = "nil" if is_nil else ""
            print(f"  → {act}: 估值={val:.1f} {penalty_str} {is_nil_str}")

        # 选择动作
        chosen = solver._rollout_select_action(sim, legal, cp, root_player)
        print(f"  选择: {chosen}")

        # 应用
        solver._apply_action_in_place(sim, chosen, cp)
        print(f"  桌面: {[(p, str(c)) for p, c in sim.table_cards]}")

        # 检查是否有完成的墩
        # trick_complete is checked inside _apply_action_in_place

    print(f"\n模拟结束!")
    print(f"tricks_won: {sim.tricks_won}")
    print(f"tricks_played: {sim.tricks_played}")

    if rules.end_trickgame(sim):
        scores = rules.score(sim)
        print(f"最终得分: {scores}")
        print(f"P0得分差: {scores[0]}")
        return scores[0]
    return None

result = trace_rollout(solver, sq_state, 0)
print(f"\n>>> SQ rollout 结果: {result}")

# 对比：出SA的rollout
sa_state = solver._apply_action(state, Card.from_str("SA"), 0)
sa_result = trace_rollout(solver, sa_state, 0)
print(f"\n>>> SA rollout 结果: {sa_result}")
