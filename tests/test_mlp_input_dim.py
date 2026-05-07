"""
测试MLP输入维度与特征编码器总维度保持一致。

该测试用于防止特征维度更新后，模型输入维度忘记同步。
运行方式:
    /mnt/c/Users/35559/Spades-AI/.venv/bin/python tests/test_mlp_input_dim.py
"""

import sys
sys.path.insert(0, '.')

from trick_taking.utils.feature_encoder import SpadesFeatureEncoder
from distill.mlp_model import DoubleDummyMLP as DistillMLP
from mlp_2_left.mlp_model import DoubleDummyMLP as TwoLeftMLP


def main():
    encoder = SpadesFeatureEncoder()
    expected_dim = encoder.total_dim

    distill_model = DistillMLP()
    two_left_model = TwoLeftMLP()

    assert distill_model.input_dim == expected_dim, (
        f"distill MLP input_dim={distill_model.input_dim} 与特征维度 {expected_dim} 不一致"
    )
    assert two_left_model.input_dim == expected_dim, (
        f"mlp_2_left MLP input_dim={two_left_model.input_dim} 与特征维度 {expected_dim} 不一致"
    )

    print("MLP输入维度一致性测试通过")
    print(f"当前特征总维度: {expected_dim}")


if __name__ == '__main__':
    main()
