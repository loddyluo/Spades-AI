"""
运行 tests/test_exact_solver_cpp_native_regression_100.py 中的回归测试（无 pytest 依赖）。
"""

from __future__ import annotations

import sys

sys.path.insert(0, '.')

from tests.test_exact_solver_cpp_native_regression_100 import test_cpp_native_vs_python_100_samples


def main():
    try:
        test_cpp_native_vs_python_100_samples()
        print("回归测试通过：100 个样本全部一致")
    except AssertionError as e:
        print("回归测试失败:", e)
        raise
    except Exception as e:
        print("回归测试运行出错:", e)
        raise


if __name__ == '__main__':
    main()
