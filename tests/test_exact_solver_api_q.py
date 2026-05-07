"""
测试精确求解器公开接口扩展：solve_with_q。

验证点：
1. solve_with_q 能返回 best_action 与 action_q_values。
2. value 与 solve(state) 一致。
3. best_action 的Q值等于当前行动方（MAX/MIN）在该节点的最优Q值。
4. 在不同剩余手牌规模（约1~25）都能稳定工作。
"""

import random
import sys
sys.path.insert(0, '.')

from trick_taking.game_state import GameState
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.utils.state_tools import create_random_state


def build_state_with_remaining_cards(target_remaining: int, seed: int) -> GameState:
    """
    从随机完整状态出发，随机推进到指定剩余手牌总数。

    参数:
        target_remaining: 目标剩余手牌总数（1~25）
        seed: 随机种子
    """
    if target_remaining < 1:
        raise ValueError("target_remaining 必须 >= 1")

    rng = random.Random(seed)
    solver = ExactDoubleDummySolver()
    state = create_random_state()

    # 使用求解器内部状态推进逻辑，保证与真实搜索状态一致。
    while sum(len(h) for h in state.hands) > target_remaining:
        pid = state.turn
        legal = solver.rules.playable(state, state.hands[pid], pid)
        if not legal:
            break
        action = rng.choice(legal)
        state = solver._apply_action(state, action, pid)

    return state


def test_state(state: GameState, tag: str) -> None:
    """单状态验证。"""
    solver = ExactDoubleDummySolver()

    value_only = solver.solve(state)
    details = solver.solve_with_q(state)

    assert abs(value_only - details["value"]) < 1e-9, f"{tag}: solve 与 solve_with_q 的值不一致"

    # 若终局/无动作，best_action 允许为 None
    q_map = details["action_q_values"]
    legal = solver.rules.playable(state, state.hands[state.turn], state.turn)

    if not legal:
        assert details["best_action"] is None, f"{tag}: 无合法动作时 best_action 应为 None"
        return

    assert len(q_map) == len(legal), f"{tag}: action_q_values 数量应等于合法动作数"
    assert details["best_action"] in legal, f"{tag}: best_action 必须属于合法动作"

    team = state.teams[state.turn]
    q_values = list(q_map.values())

    if team == 0:
        optimal_q = max(q_values)
    else:
        optimal_q = min(q_values)

    best_q = q_map[details["best_action"]]
    assert abs(best_q - optimal_q) < 1e-9, f"{tag}: best_action 对应Q值不是最优"


if __name__ == "__main__":
    # 覆盖 1~25 附近不同规模
    targets = [1, 5, 9, 13, 17, 21, 25]
    for i, target in enumerate(targets):
        s = build_state_with_remaining_cards(target, seed=1000 + i)
        total = sum(len(h) for h in s.hands)
        test_state(s, tag=f"remain={total}")

    print("test_exact_solver_api_q: 所有测试通过")
