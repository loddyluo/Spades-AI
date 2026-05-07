"""
对比原始 Python 参考实现与 C++ 原生实现的正确性回归（100 个确定性样本）。

说明：
- 只测试小牌数情形（使用已有的 generate_2card_state 来构造极小牌局），以便快速遍历大量样本。
- 对每个样本，逐项比较 `solve_with_q` 返回的 `value`、`best_action` 和 `action_q_values`。
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, '.')

from mlp_2_left.generate_2card_states import generate_2card_state
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummyPythonSolver
from trick_taking.solvers.exact_double_dummy_cpp_native import ExactDoubleDummyCppNativeSolver


def assert_q_dicts_equal(d1, d2):
    """断言两个 action->q 字典在键和值上等价（浮点数使用 isclose）。"""
    assert set(d1.keys()) == set(d2.keys()), f"动作集合不一致: {set(d1.keys())} vs {set(d2.keys())}"
    for k in d1:
        v1 = d1[k]
        v2 = d2[k]
        if not math.isclose(v1, v2, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"Q 值不一致: action={k}, py={v1}, cpp={v2}")


def test_cpp_native_vs_python_100_samples():
    """使用 100 个不同 seed 的 2-card 状态进行逐项对比。"""
    py_solver = ExactDoubleDummyPythonSolver()
    cpp_solver = ExactDoubleDummyCppNativeSolver()

    # 必须确保 C++ 原生库可用
    if not cpp_solver.native_available:
        import pytest

        pytest.skip("C++ 原生实现不可用，跳过本回归测试")

    for seed in range(100):
        state = generate_2card_state(seed=seed)

        py_ret = py_solver.solve_with_q(state)
        cpp_ret = cpp_solver.solve_with_q(state)

        # 值与最优动作必须一致
        assert math.isclose(py_ret["value"], cpp_ret["value"], rel_tol=1e-9, abs_tol=1e-9), (
            f"value mismatch seed={seed}: py={py_ret['value']} cpp={cpp_ret['value']}"
        )

        # best_action 可能为 None 或 Card，直接比较
        assert py_ret["best_action"] == cpp_ret["best_action"], (
            f"best_action mismatch seed={seed}: py={py_ret['best_action']} cpp={cpp_ret['best_action']}"
        )

        # action_q_values 字典逐项比较
        assert_q_dicts_equal(py_ret["action_q_values"], cpp_ret["action_q_values"])
