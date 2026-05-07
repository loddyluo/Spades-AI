"""
调试精确求解器：去掉缓存，追踪SQ路径的minimax值
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trace_exact2 import create_two_trick_state

# Patch: 创建一个去掉缓存的求解器
class NoCacheExactSolver(ExactDoubleDummySolver):
    def _minimax(self, state, alpha, beta):
        if self.rules.end_trickgame(state):
            val = self._compute_score_diff(state)
            return val

        # NO CACHE CHECK

        current_player = state.turn
        hand = state.hands[current_player]
        legal_actions = self.rules.playable(state, hand, current_player)

        if not legal_actions:
            return self._compute_score_diff(state)

        current_team = state.teams[current_player]

        depth = 13 - state.tricks_played

        if current_team == 0:
            value = -float('inf')
            for action in legal_actions:
                new_state = self._apply_action(state, action, current_player)
                child_value = self._minimax(new_state, alpha, beta)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if value >= beta:
                    break
        else:
            value = float('inf')
            for action in legal_actions:
                new_state = self._apply_action(state, action, current_player)
                child_value = self._minimax(new_state, alpha, beta)
                value = min(value, child_value)
                beta = min(beta, value)
                if value <= alpha:
                    break

        return value


state = create_two_trick_state()
rules = SpadesRules()

print("使用无缓存求解器（避免任何缓存相关bug）...")
solver = NoCacheExactSolver()
root_val = solver.solve(state)
print(f"根节点最优值: {root_val}")

# 测试每个第一动作
hand0 = state.hands[0]
for action in rules.playable(state, hand0, 0):
    new_state = solver._apply_action(state, action, 0)
    fresh_solver = NoCacheExactSolver()
    value = fresh_solver.solve(new_state)
    print(f"  出 {action}: 后续最优值 = {value}")

# 额外：针对SQ路线，输出每个子状态的终端值
print("\n详细追踪SQ路线...")
sq_state = solver._apply_action(state, Card.from_str("SQ"), 0)

# P1必须出SK
sk_state = solver._apply_action(sq_state, Card.from_str("SK"), 1)

# P2可以选择H2, HT, HJ
for p2card_str in ["H2", "HT", "HJ"]:
    p2card = Card.from_str(p2card_str)
    p2_state = solver._apply_action(sk_state, p2card, 2)

    for p3card_str in ["H6", "H8", "H9"]:
        p3card = Card.from_str(p3card_str)
        p3_state = solver._apply_action(p2_state, p3card, 3)

        # 谜1后：P1赢
        print(f"\nSQ->SK->{p2card_str}->{p3card_str}: P1赢, tricks_played={p3_state.tricks_played}, tricks_won={p3_state.tricks_won}")
        print(f"  P0手牌: {[str(c) for c in p3_state.hands[0]]}")
        print(f"  P1手牌: {[str(c) for c in p3_state.hands[1]]}")
        print(f"  P2手牌: {[str(c) for c in p3_state.hands[2]]}")
        print(f"  P3手牌: {[str(c) for c in p3_state.hands[3]]}")

        # 用无缓存求解器评估这个子状态
        sub_solver = NoCacheExactSolver()
        val = sub_solver.solve(p3_state)
        print(f"  子状态minimax值: {val}")

        # 手动计算如果P1全赢的终局价值
        # 模拟P1一直赢到结束
        from copy import deepcopy
        sim = deepcopy(p3_state)
        # 继续模拟完
        while not rules.end_trickgame(sim):
            cp = sim.turn
            hand = sim.hands[cp]
            legal = rules.playable(sim, hand, cp)
            if not legal:
                break
            # 总是用第一个合法动作
            act = legal[0]
            sim.play_card_to_table(cp, act)
            if len(sim.table_cards) == 4:
                spades = [(p,c) for p,c in sim.table_cards if c.suit == Suit.SPADES]
                if spades:
                    winner, _ = max(spades, key=lambda x: x[1].rank.value)
                else:
                    lead_suit = sim.table_cards[0][1].suit
                    suit_cards = [(p,c) for p,c in sim.table_cards if c.suit == lead_suit]
                    winner, _ = max(suit_cards, key=lambda x: x[1].rank.value)
                sim.complete_trick(winner)
                sim.trick_leader = winner
                sim.turn = winner

        scores = rules.score(sim)
        print(f"  模拟终局 tricks_won={sim.tricks_won}, P0得分差={scores[0]}")
