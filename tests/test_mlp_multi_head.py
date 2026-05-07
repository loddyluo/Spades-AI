"""测试新的双头 MLP 训练链路。"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, '.')

from data.training_data import load_bucket_dataset
from mlp.mlp_model import DoubleDummyMLP
from mlp.training_utils import (
    build_policy_target,
    masked_policy_accuracy,
    masked_policy_loss,
    prepare_multi_head_arrays,
)


def build_fake_sample() -> dict:
    """构造一个最小的训练样本。"""
    feature = torch.linspace(0.0, 1.0, 1229, dtype=torch.float32)
    action_ids = torch.tensor([3, 7, 12], dtype=torch.int64)
    action_q_values = torch.tensor([10.0, 30.0, -5.0], dtype=torch.float32)

    return {
        "x": 24,
        "seed": 123,
        "feature": feature,
        "value_view": torch.tensor(50.0, dtype=torch.float32),
        "value_team0": torch.tensor(50.0, dtype=torch.float32),
        "best_action_id": 7,
        "current_player": 0,
        "optimize_for_team": 0,
        "action_ids": action_ids,
        "action_q_values": action_q_values,
        "feature_dim": 1229,
    }


def test_policy_target_builder() -> None:
    policy_target, policy_mask = build_policy_target([3, 7, 12], [10.0, 30.0, -5.0])

    assert policy_target.shape == (52,)
    assert policy_mask.shape == (52,)
    assert np.isclose(policy_target.sum(), 1.0)
    assert policy_target[3] > 0.0
    assert policy_target[7] > policy_target[3]
    assert policy_target[12] > 0.0
    assert policy_target[0] == 0.0
    assert policy_mask[7] == 1.0
    assert policy_mask[0] == 0.0


def test_prepare_arrays_and_shapes() -> None:
    sample = build_fake_sample()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "fake.pt"
        torch.save({"meta": {"num_samples": 1}, "samples": [sample]}, tmp_path)

        loaded = load_bucket_dataset(tmp_path)
        arrays = prepare_multi_head_arrays(loaded["samples"])

    assert arrays["features"].shape == (1, 1229)
    assert arrays["value_targets"].shape == (1, 1)
    assert arrays["policy_targets"].shape == (1, 52)
    assert arrays["policy_masks"].shape == (1, 52)
    assert arrays["best_action_ids"].shape == (1,)
    assert np.isclose(arrays["value_targets"][0, 0], 2.0)
    assert np.isclose(arrays["policy_targets"][0].sum(), 1.0)


def test_forward_loss_and_backward() -> None:
    model = DoubleDummyMLP(input_dim=1229, hidden_dims=[64, 32], policy_output_dim=52)

    batch = prepare_multi_head_arrays([build_fake_sample(), build_fake_sample()])
    X = torch.from_numpy(batch["features"])
    value_targets = torch.from_numpy(batch["value_targets"])
    policy_targets = torch.from_numpy(batch["policy_targets"])
    policy_masks = torch.from_numpy(batch["policy_masks"])
    best_action_ids = torch.from_numpy(batch["best_action_ids"])

    outputs = model(X)
    assert outputs["value"].shape == (2, 1)
    assert outputs["policy_logits"].shape == (2, 52)

    value_loss = torch.nn.functional.mse_loss(outputs["value"], value_targets)
    policy_loss = masked_policy_loss(outputs["policy_logits"], policy_targets, policy_masks)
    total_loss = value_loss + policy_loss

    assert torch.isfinite(total_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    policy_acc = masked_policy_accuracy(outputs["policy_logits"], policy_masks, best_action_ids)
    assert 0.0 <= float(policy_acc.item()) <= 1.0


def main() -> None:
    test_policy_target_builder()
    test_prepare_arrays_and_shapes()
    test_forward_loss_and_backward()
    print("双头 MLP 测试通过")


if __name__ == "__main__":
    main()
