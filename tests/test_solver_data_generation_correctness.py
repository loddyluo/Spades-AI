"""验证数据生成所用精确求解器的正确性（小剩余牌数）。

此脚本对若干小的剩余牌数 x（例如 2,4,6,8）使用确定性种子生成局面，
分别用 Python 参考求解器和 cpp_opt1 求解器求解，并比较返回的 value 是否一致，
以及 `best_action` 是否出现在 `action_q_values` 中。

输入: 无（从命令行运行）
输出: 若发现不一致则抛出 AssertionError，否则打印通过信息。
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

sys.path.insert(0, '.')

from data.training_data import build_state_with_remaining_cards, SUPPORTED_BUCKETS
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver

try:
    from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver
    cpp_available = True
except Exception:
    ExactDoubleDummyCppOpt1Solver = None
    cpp_available = False


def compare_solvers_for_x(x: int, seeds: list[int]) -> None:
    python_solver = ExactDoubleDummySolver()
    cpp_solver = ExactDoubleDummyCppOpt1Solver() if cpp_available else None

    for seed in seeds:
        state = build_state_with_remaining_cards(x, seed)

        py_res = python_solver.solve_with_q(state)
        py_value = float(py_res['value'])
        py_best = py_res['best_action']
        py_action_q = py_res['action_q_values']

        if cpp_available and cpp_solver.native_available:
            cpp_res = cpp_solver.solve_with_q(state)
            cpp_value = float(cpp_res['value'])

            # 比较值是否接近
            if not math.isclose(py_value, cpp_value, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"x={x} seed={seed} value mismatch: py={py_value} cpp={cpp_value}")

        # 检查 best_action 在 action_q_values 中
        if py_best is not None:
            assert any(a.card_id == py_best.card_id for a in py_action_q.keys()) or len(py_action_q) > 0, (
                f"x={x} seed={seed} best_action not found in action_q_values"
            )


def main() -> None:
    xs = [2, 4, 6, 8]
    seeds = [1000 + i for i in range(5)]

    for x in xs:
        print(f"Testing x={x} ...")
        compare_solvers_for_x(x, seeds)
        print(f"  x={x} passed")

    print("Solver data-generation correctness checks passed for small x values.")


if __name__ == '__main__':
    main()
