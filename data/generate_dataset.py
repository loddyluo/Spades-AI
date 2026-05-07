"""批量生成训练数据。

默认按 x=24/25/28/32 分桶，并分别保存成独立的 PyTorch 文件。

示例:
    /mnt/c/Users/35559/Spades-AI/.venv/bin/python data/generate_dataset.py --xs 24 25 28 32 --num_samples 1000 --output_dir data
 
模块函数说明（输入/输出）:

- main() -> None
    命令行入口，参数包括 --xs (list of int), --num_samples, --output_dir, --prefix, --benchmark
    输出: 将生成的数据保存为 PyTorch .pt 文件，位置由 --output_dir 指定
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '.')

from data.training_data import (
    SUPPORTED_BUCKETS,
    benchmark_generation,
    dataset_path,
    generate_bucket_dataset,
    save_bucket_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xs", type=int, nargs="+", default=list(SUPPORTED_BUCKETS), help="要生成的剩余牌数桶")
    parser.add_argument("--num_samples", type=int, default=1000, help="每个桶生成多少条数据")
    parser.add_argument("--seed_start", type=int, default=0, help="起始种子")
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    parser.add_argument("--prefix", type=str, default="spades_dd", help="文件名前缀")
    parser.add_argument("--benchmark", action="store_true", help="只打印耗时，不保存文件")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for x in args.xs:
        if args.benchmark:
            report = benchmark_generation(x, args.num_samples, seed_start=args.seed_start)
            print(
                f"x={x} | num_samples={report['num_samples']} | elapsed={report['elapsed']:.3f}s | avg={report['avg']:.4f}s"
            )
            continue

        samples = generate_bucket_dataset(x, args.num_samples, seed_start=args.seed_start)
        out_path = dataset_path(output_dir, x, args.num_samples, prefix=args.prefix)
        save_bucket_dataset(samples, out_path)
        print(f"已保存: {out_path} | 样本数={len(samples)}")


if __name__ == "__main__":
    main()
