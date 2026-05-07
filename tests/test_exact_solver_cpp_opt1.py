"""
opt1 新优化尝试的正确性测试。

说明：
- 该测试不修改现有正确求解器，只对比新尝试 `ExactDoubleDummyCppOpt1Solver` 与
  Python 参考实现的一致性。
- 重点覆盖小牌数和大量样本。
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, '.')

from mlp_2_left.generate_2card_states import generate_2card_state
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummyPythonSolver
from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver


def test_opt1_matches_python_on_100_small_states():
    """100 个小牌样本：逐项比较 value / best_action / action_q_values。"""
    base = ExactDoubleDummyPythonSolver()
    opt1 = ExactDoubleDummyCppOpt1Solver()

    if not opt1.native_available:
        raise RuntimeError("opt1 原生库不可用，无法执行正确性测试")

    for seed in range(100):
        s = generate_2card_state(seed=seed)
        b = base.solve_with_q(s)
        o = opt1.solve_with_q(s)

        assert math.isclose(b["value"], o["value"], rel_tol=1e-9, abs_tol=1e-9), (
            f"value mismatch seed={seed}: {b['value']} vs {o['value']}"
        )
        assert b["best_action"] == o["best_action"], (
            f"best_action mismatch seed={seed}: {b['best_action']} vs {o['best_action']}"
        )

        assert set(b["action_q_values"].keys()) == set(o["action_q_values"].keys()), (
            f"action keys mismatch seed={seed}"
        )
        for a in b["action_q_values"]:
            assert math.isclose(b["action_q_values"][a], o["action_q_values"][a], rel_tol=1e-9, abs_tol=1e-9), (
                f"q mismatch seed={seed}, action={a}: {b['action_q_values'][a]} vs {o['action_q_values'][a]}"
            )


if __name__ == '__main__':
    test_opt1_matches_python_on_100_small_states()
    print("test_exact_solver_cpp_opt1: 所有测试通过")
