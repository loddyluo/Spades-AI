"""测试用的数据生成速度程序。

默认对 x=24/28/32 各生成 20 条样本，并输出耗时。
 
模块函数说明（输入/输出）:

- main() -> None
    无输入（通过命令行执行），会对 SUPPORTED_BUCKETS 各生成 20 条样本并打印耗时报告。
"""

from __future__ import annotations

import sys

sys.path.insert(0, '.')

from data.training_data import SUPPORTED_BUCKETS, benchmark_generation


def main() -> None:
    print("=" * 100)
    print("训练数据生成速度测试：x=24/28/32，每个桶20条")
    print("=" * 100)

    for x in SUPPORTED_BUCKETS:
        report = benchmark_generation(x, 20, seed_start=1000 + x * 100)
        print(
            f"x={x:2d} | num_samples={report['num_samples']:2d} | total={report['elapsed']:.3f}s | avg={report['avg']:.4f}s"
        )

    print("-" * 100)
    print("速度测试完成。")


if __name__ == "__main__":
    main()
