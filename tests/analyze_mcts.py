"""
分析MCTS为什么找不到SQ路线（最优值120）

问题发现：
- `_determine_trick_winner` 正确处理了将牌规则（黑桃在任何情况下都是将牌）
- 但 rollout 的贪心策略（用 `_rollout_state_value` 做一步前瞻）会误判
- P0在梅花领出的墩上如果出SA（将牌），立刻赢墩，状态评分为120（看起来很好）
- 但实际上P0赢了这个墩后，下一墩领出H3，P2会赢墩（nil失败），最终得分只有11
- 一步前瞻的评分函数被误导了，因为它只看当前状态的得分，不看后续发展

MCTS收敛问题：SQ的初始rollout值很低（~11），UCT不会优先探索它
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.solvers.double_dummy import DoubleDummySolver
from trace_exact2 import create_two_trick_state

# 验证正确值
state = create_two_trick_state()
exact = ExactDoubleDummySolver()
print(f"精确求解器值: {exact.solve(state)}")

# 测试不同参数的MCTS
for n_iter in [10000, 20000, 50000, 100000]:
    for eps in [0.0, 0.05, 0.1, 0.2]:
        solver = DoubleDummySolver(
            max_iterations=n_iter,
            exploration_weight=1.4,
            rollout_epsilon=eps
        )
        result = solver.solve(state, current_player=0)
        score = result['state_evaluation']['expected_score_diff']
        best = str(result['best_action'])
        print(f"iter={n_iter:6d} eps={eps:.2f}: 最优={best}, 价值={score:.1f}")
        if abs(score - 120.0) < 1:
            print(f"  ^^^ 找到最优解! ^^^")
            break
    else:
        continue
    break

# 用更高的探索权重测试
print("\n高探索权重测试:")
for C in [1.4, 2.0, 5.0, 10.0]:
    solver = DoubleDummySolver(
        max_iterations=20000,
        exploration_weight=C,
        rollout_epsilon=0.0
    )
    result = solver.solve(state, current_player=0)
    score = result['state_evaluation']['expected_score_diff']
    best = str(result['best_action'])
    print(f"  C={C:.1f}: 最优={best}, 价值={score:.1f}")
