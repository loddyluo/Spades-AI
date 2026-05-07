"""统计训练数据中的重复率与特征相似度。

文件作用：
- 读取指定数据集，统计 `feature` 的完全重复率；
- 使用随机抽样的样本对，统计特征余弦相似度分布；
- 帮助判断大规模生成时是否出现大量重复或高度相似局面。

输入/输出说明：
- load_dataset(dataset_path: Path) -> list[dict]
  输入: PyTorch 数据文件路径。
  输出: 样本列表，每个样本至少包含 `feature` 字段。

- compute_duplicate_rate(samples: list[dict]) -> dict[str, float]
  输入: 样本列表。
  输出: 重复统计，包含 total / unique / duplicate / duplicate_rate。

- sample_feature_similarity(features: np.ndarray, num_pairs: int, seed: int) -> dict[str, float]
  输入: feature 矩阵、抽样对数、随机种子。
  输出: 余弦相似度统计，包含 mean / std / p50 / p90 / p95 / p99 / high_sim_rate。

- main() -> None
  输入: 命令行参数 `--dataset`、`--num_pairs`、`--high_sim_threshold`。
  输出: 打印统计结果。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from data.training_data import load_bucket_dataset


def load_dataset(dataset_path: Path) -> list[dict]:
    """读取数据集中的样本列表。"""
    loaded = load_bucket_dataset(dataset_path)
    samples = loaded.get("samples", [])
    if not isinstance(samples, list):
        raise TypeError("数据文件中的 samples 字段必须是列表")
    return samples


def _feature_signature(feature: np.ndarray) -> str:
    """为单条 feature 生成稳定签名。"""
    contiguous = np.ascontiguousarray(feature, dtype=np.float32)
    return hashlib.sha1(contiguous.tobytes()).hexdigest()


def compute_duplicate_rate(samples: list[dict]) -> dict[str, float]:
    """统计完全重复的 feature 比例。"""
    signatures: set[str] = set()
    duplicate_count = 0

    for sample in samples:
        feature = np.asarray(sample["feature"], dtype=np.float32)
        signature = _feature_signature(feature)
        if signature in signatures:
            duplicate_count += 1
        else:
            signatures.add(signature)

    total = len(samples)
    unique = len(signatures)
    duplicate_rate = duplicate_count / total if total > 0 else 0.0
    return {
        "total": float(total),
        "unique": float(unique),
        "duplicate": float(duplicate_count),
        "duplicate_rate": float(duplicate_rate),
    }


def sample_feature_similarity(features: np.ndarray, num_pairs: int, seed: int, high_sim_threshold: float) -> dict[str, float]:
    """随机抽样样本对，统计余弦相似度分布。"""
    if features.ndim != 2:
        raise ValueError("features 必须是二维数组")
    if len(features) < 2:
        raise ValueError("features 至少需要两条样本")

    rng = np.random.default_rng(seed)
    n = len(features)
    left_indices = rng.integers(0, n, size=num_pairs)
    right_indices = rng.integers(0, n, size=num_pairs)
    same_mask = left_indices == right_indices
    if np.any(same_mask):
        right_indices[same_mask] = (right_indices[same_mask] + 1) % n

    left = features[left_indices].astype(np.float32, copy=False)
    right = features[right_indices].astype(np.float32, copy=False)

    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    valid = denom > 0
    cosine = np.zeros(num_pairs, dtype=np.float32)
    if np.any(valid):
        cosine[valid] = np.sum(left[valid] * right[valid], axis=1) / denom[valid]

    return {
        "mean": float(np.mean(cosine)),
        "std": float(np.std(cosine)),
        "p50": float(np.percentile(cosine, 50)),
        "p90": float(np.percentile(cosine, 90)),
        "p95": float(np.percentile(cosine, 95)),
        "p99": float(np.percentile(cosine, 99)),
        "high_sim_rate": float(np.mean(cosine >= high_sim_threshold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/spades_dd_x24_n100000.pt",
        help="要检查的数据文件",
    )
    parser.add_argument("--num_pairs", type=int, default=20000, help="抽样相似度对数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--high_sim_threshold", type=float, default=0.995, help="高相似阈值")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = load_dataset(dataset_path)
    if not samples:
        raise RuntimeError("数据集为空")

    duplicate_stats = compute_duplicate_rate(samples)
    features = np.asarray([np.asarray(sample["feature"], dtype=np.float32) for sample in samples], dtype=np.float32)
    similarity_stats = sample_feature_similarity(features, args.num_pairs, args.seed, args.high_sim_threshold)

    print("=" * 100)
    print("训练数据重复率与特征相似度统计")
    print("=" * 100)
    print(f"数据文件: {dataset_path}")
    print(f"样本数: {int(duplicate_stats['total'])}")
    print(f"完全重复数: {int(duplicate_stats['duplicate'])}")
    print(f"唯一样本数: {int(duplicate_stats['unique'])}")
    print(f"完全重复率: {duplicate_stats['duplicate_rate']:.8f}")
    print("-" * 100)
    print(f"抽样对数: {args.num_pairs}")
    print(f"余弦相似度均值: {similarity_stats['mean']:.6f}")
    print(f"余弦相似度标准差: {similarity_stats['std']:.6f}")
    print(f"p50: {similarity_stats['p50']:.6f}")
    print(f"p90: {similarity_stats['p90']:.6f}")
    print(f"p95: {similarity_stats['p95']:.6f}")
    print(f"p99: {similarity_stats['p99']:.6f}")
    print(f"高相似阈值({args.high_sim_threshold})比例: {similarity_stats['high_sim_rate']:.6f}")


if __name__ == "__main__":
    main()