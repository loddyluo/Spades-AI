"""
测试原生 C++ 精确双明手求解器的正确性。

验证点：
1. native 版本与 Python 参考版本在同一状态上 solve 值一致。
2. native 的 solve_with_q 与 Python 参考版本的动作Q值逐项一致。
3. 覆盖 16 张及以下的随机样本，控制正确性测试耗时。
"""

import random
import sys
sys.path.insert(0, '.')

from mlp_2_left.generate_2card_states import generate_2card_state
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummyPythonSolver
from trick_taking.solvers.exact_double_dummy_cpp_native import ExactDoubleDummyCppNativeSolver
from trick_taking.utils.state_tools import create_random_state


def build_state_with_remaining_cards(target_remaining: int, seed: int):
    rng = random.Random(seed)
    helper = ExactDoubleDummyPythonSolver()
    saved_state = random.getstate()
    random.seed(seed)
    state = create_random_state()
    random.setstate(saved_state)
    while sum(len(h) for h in state.hands) > target_remaining:
        pid = state.turn
        legal = helper.rules.playable(state, state.hands[pid], pid)
        if not legal:
            break
        state = helper._apply_action(state, rng.choice(legal), pid)
    return state


def assert_same(state, tag: str):
    base = ExactDoubleDummyPythonSolver()
    native = ExactDoubleDummyCppNativeSolver()

    v1 = base.solve(state)
    v2 = native.solve(state)
    assert abs(v1 - v2) < 1e-9, f"{tag}: 值不一致 base={v1}, native={v2}"

    details = native.solve_with_q(state)
    assert abs(details["value"] - v2) < 1e-9, f"{tag}: solve_with_q 与 solve 不一致"

    base_q = base.solve_with_q(state)
    assert abs(base_q["value"] - details["value"]) < 1e-9, f"{tag}: root value 不一致"
    assert set(base_q["action_q_values"].keys()) == set(details["action_q_values"].keys()), f"{tag}: 动作集合不一致"
    for action, q_value in base_q["action_q_values"].items():
        assert abs(q_value - details["action_q_values"][action]) < 1e-9, f"{tag}: 动作Q值不一致 {action}"


if __name__ == "__main__":
    assert_same(generate_2card_state(seed=31), "2card")

    fixed_points = [1, 2, 3, 5, 8, 10, 13, 15, 16]
    for i, t in enumerate(fixed_points):
        s = build_state_with_remaining_cards(t, seed=5000 + i)
        assert_same(s, f"remain={sum(len(h) for h in s.hands)}")

    # 附加随机回归：补充多样局面，同时控制总耗时。
    for i in range(4):
        remain = random.randint(1, 16)
        s = build_state_with_remaining_cards(remain, seed=9000 + i)
        assert_same(s, f"random_remain={sum(len(h) for h in s.hands)}")

    print("test_exact_solver_cpp_native: 所有测试通过")
