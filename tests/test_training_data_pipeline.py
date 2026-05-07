"""训练数据管线格式测试。

验证点：
1. 生成的样本满足 x=24/28/32 的剩余牌数要求。
2. feature 维度与 feature_encoder 一致。
3. action_q_values / best_action / value 字段齐全，且能被 torch.save / torch.load 正常处理。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import torch

sys.path.insert(0, '.')

from data.training_data import (
    SUPPORTED_BUCKETS,
    dataset_path,
    generate_bucket_dataset,
    load_bucket_dataset,
    save_bucket_dataset,
)
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def main() -> None:
    encoder = SpadesFeatureEncoder()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for x in SUPPORTED_BUCKETS:
            samples = generate_bucket_dataset(x, 2, seed_start=7000 + x)
            assert len(samples) == 2

            for sample in samples:
                assert sample["x"] == x
                assert sample["feature"].shape[0] == encoder.total_dim
                assert sample["feature_dim"] == encoder.total_dim
                assert sample["action_ids"].ndim == 1
                assert sample["action_q_values"].ndim == 1
                assert len(sample["action_ids"]) == len(sample["action_q_values"])
                assert sample["value_team0"].ndim == 0
                assert sample["value_view"].ndim == 0
                assert sample["best_action_id"] == -1 or sample["best_action_id"] in sample["action_ids"].tolist()

            out_file = dataset_path(tmp_dir, x, len(samples), prefix="unit_test")
            save_bucket_dataset(samples, out_file)
            assert out_file.exists()

            loaded = load_bucket_dataset(out_file)
            assert loaded["meta"]["num_samples"] == len(samples)
            assert loaded["meta"]["feature_dim"] == encoder.total_dim
            assert len(loaded["samples"]) == len(samples)
            assert loaded["samples"][0]["x"] == x

            # 额外检查：PyTorch 直接读取没有问题。
            raw = torch.load(out_file, map_location="cpu")
            assert raw["meta"]["num_samples"] == len(samples)

    print("训练数据管线格式测试通过")


if __name__ == "__main__":
    main()
