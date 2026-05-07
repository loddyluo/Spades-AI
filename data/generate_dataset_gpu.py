"""GPU 感知的并行数据生成器。

目标：保持与 `data.generate_dataset.py` 完全相同的生成逻辑与种子确定性，
但在生成时通过多进程并行化 CPU-bound 的求解器调用，并在主进程中把特征批量移动到 GPU（若可用），以利用 GPU 做后续可能的张量操作。

注意：当前精确求解器为 C++/CPU 实现，无法用 GPU 加速。此脚本通过并行化 solver 调用与可选的 feature 张量化到 GPU 来加速总体流水线（但并不能把 solver 移到 GPU）。

用法示例:
    python data/generate_dataset_gpu.py --xs 24 --num_samples 1000 --output_dir data --prefix spades_dd_gpu --num_workers 4

模块接口说明：
- main() -> None: 命令行入口，参数包括 --xs, --num_samples, --seed_start, --output_dir, --prefix, --num_workers
    输出: 保存与 `data/generate_dataset.py` 相同格式的 .pt 文件（meta + samples）

实现要点：
- 使用 `concurrent.futures.ProcessPoolExecutor` 并发运行单样本生成函数（每个进程会独立创建 encoder 与 solver，保证确定性）
- 收集结果后按 seed 排序以保证输出顺序与串行版本一致
- 若 GPU 可用，会把所有 sample 的 `feature` 批量堆叠为 Tensor 并移动到 `cuda`（然后再移回 cpu 保存），以演示和利用可用 GPU
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

# 允许脚本直接运行时能正确导入本仓库的模块
sys.path.insert(0, '.')

from data.training_data import dataset_path, generate_bucket_sample, dataset_file_name, SUPPORTED_BUCKETS


def _generate_single(args: tuple[int, int]) -> tuple[int, dict[str, Any]]:
    """在子进程里生成单个样本。

    返回 (seed, sample_dict)。为避免在进程间传递 `torch.Tensor` 导致使用 FD-based
    reductions 出错，这里把所有 `torch.Tensor` 转成 Python 原生可序列化的对象
   （list 或标量）。父进程收到后会再把其恢复为 `torch.Tensor`。

    该函数顶层可被 multiprocessing 导入执行。
    """
    import numpy as _np
    target_remaining, seed = args
    sample = generate_bucket_sample(target_remaining, seed)

    # 将 torch.Tensor 转为 Python 原生类型，避免共享 FD 的 pickling
    for k, v in list(sample.items()):
        try:
            # 仅对 torch.Tensor 做处理
            import torch as _torch

            if isinstance(v, _torch.Tensor):
                # 把 tensor 转成 numpy，再转成 list（尽量保证可序列化）
                arr = v.cpu().numpy()
                # 对一维整型/浮点向量，转为 list；对标量也转为 Python 标量
                if arr.ndim == 0:
                    sample[k] = arr.item()
                else:
                    sample[k] = arr.tolist()
        except Exception:
            # 如果任何步骤失败，保持原样（将会由父进程捕获序列化错误）
            pass

    return seed, sample


def generate_bucket_dataset_parallel(target_remaining: int, num_samples: int, *, seed_start: int = 0, num_workers: int | None = None) -> list[dict[str, Any]]:
    """并行生成样本，按 seed 顺序返回样本列表。

    - 每个子进程会独立运行 `generate_bucket_sample` 保证与原逻辑一致。
    - 主进程会在收集完所有样本后把 `feature` 批量堆叠并移动到 GPU（若可用），随后再移回 CPU 保存，保证输出与原处于相同 dtype 与 shape。
    """
    if target_remaining not in SUPPORTED_BUCKETS:
        raise ValueError(f"不支持的桶: x={target_remaining}, 仅支持 {SUPPORTED_BUCKETS}")

    if num_workers is None or num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    seeds = [seed_start + i for i in range(num_samples)]
    samples_by_seed: dict[int, dict[str, Any]] = {}

    # 提交到进程池并行计算
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_generate_single, (target_remaining, s)): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                s, sample = fut.result()
            except Exception as e:
                raise RuntimeError(f"生成 seed={seed} 失败: {e}")
            samples_by_seed[s] = sample

    # 按 seed 顺序收集并在父进程中把序列化的字段恢复为 torch.Tensor
    ordered_raw = [samples_by_seed[s] for s in seeds]
    ordered: list[dict[str, Any]] = []
    import numpy as _np
    for raw in ordered_raw:
        sample = dict(raw)  # shallow copy
        # 恢复常见的 tensor 字段
        if isinstance(sample.get("feature"), list):
            sample["feature"] = torch.tensor(sample["feature"], dtype=torch.float32)
        if isinstance(sample.get("action_ids"), list):
            sample["action_ids"] = torch.tensor(sample["action_ids"], dtype=torch.int64)
        if isinstance(sample.get("action_q_values"), list):
            sample["action_q_values"] = torch.tensor(sample["action_q_values"], dtype=torch.float32)
        if isinstance(sample.get("value_team0"), (float, int)):
            sample["value_team0"] = torch.tensor(float(sample["value_team0"]), dtype=torch.float32)
        if isinstance(sample.get("value_view"), (float, int)):
            sample["value_view"] = torch.tensor(float(sample["value_view"]), dtype=torch.float32)

        ordered.append(sample)

    # 将 feature 批量移动到 GPU（若可用），以便在有大批量张量处理时利用 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        feats = torch.stack([s["feature"] for s in ordered], dim=0)
        feats = feats.to(device)
        # 目前没有额外在 GPU 上的计算需求；为了保证保存时数据回到 CPU，我们再移回
        feats = feats.to("cpu")
        for i, s in enumerate(ordered):
            s["feature"] = feats[i]

    return ordered


def save_bucket_dataset(samples: list[dict[str, Any]], output_path: str | Path) -> None:
    import time
    import torch as _torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "cpp_opt1_parallel_gpu",
        "feature_dim": int(samples[0]["feature_dim"]) if samples else None,
        "supported_buckets": list(SUPPORTED_BUCKETS),
        "num_samples": len(samples),
    }
    _torch.save({"meta": meta, "samples": samples}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xs", type=int, nargs="+", default=[24], help="要生成的剩余牌数桶")
    parser.add_argument("--num_samples", type=int, default=1000, help="每个桶生成多少条数据")
    parser.add_argument("--seed_start", type=int, default=0, help="起始种子")
    parser.add_argument("--output_dir", type=str, default="data", help="输出目录")
    parser.add_argument("--prefix", type=str, default="spades_dd_gpu", help="文件名前缀")
    parser.add_argument("--num_workers", type=int, default=None, help="进程池大小 (默认: cpu_count-1)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for x in args.xs:
        samples = generate_bucket_dataset_parallel(x, args.num_samples, seed_start=args.seed_start, num_workers=args.num_workers)
        out_path = dataset_path(out_dir, x, args.num_samples, prefix=args.prefix)
        save_bucket_dataset(samples, out_path)
        print(f"已保存: {out_path} | 样本数={len(samples)}")


if __name__ == "__main__":
    main()
