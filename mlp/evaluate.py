"""评估新的双头 MLP 模型。

默认按 x=24/28/32 分桶读取数据，输出每个桶的 value 指标和 policy 指标。
 
模块函数说明（输入/输出）:

- evaluate_bucket(model: DoubleDummyMLP, dataset_path: Path) -> dict
    输入: model 实例, dataset_path 指向单个 .pt 数据文件
    输出: dict, 包含 value / policy 指标 (mae, rmse, corr, sign_acc, policy_loss, policy_acc, count)

- evaluate_bucket_files(model, dataset_files: list[Path]) -> dict
    输入: model, 一组数据文件路径
    输出: 合并评估指标字典

- main() -> None
    命令行入口，解析参数并对指定桶进行评估。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import torch

from data.training_data import DEFAULT_OUTPUT_PREFIX, SUPPORTED_BUCKETS, load_bucket_dataset
from mlp.mlp_model import DoubleDummyMLP
from mlp.training_utils import masked_policy_accuracy, masked_policy_loss, prepare_multi_head_arrays
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def evaluate_bucket(model: DoubleDummyMLP, dataset_path: Path) -> dict[str, float]:
    loaded = load_bucket_dataset(dataset_path)
    arrays = prepare_multi_head_arrays(loaded["samples"])

    features = arrays["features"]
    value_targets = arrays["value_targets"]
    policy_targets = arrays["policy_targets"]
    policy_masks = arrays["policy_masks"]
    best_action_ids = arrays["best_action_ids"]

    value_preds = []
    policy_logits_list = []
    for feature in features:
        outputs = model(feature)
        value_preds.append(float(outputs["value"].squeeze().item()))
        policy_logits_list.append(outputs["policy_logits"].squeeze(0).cpu().numpy())

    value_targets_arr = value_targets.squeeze(-1).astype(np.float64)
    value_preds_arr = np.array(value_preds, dtype=np.float64)
    policy_logits_arr = np.array(policy_logits_list, dtype=np.float32)

    mae = float(np.mean(np.abs(value_preds_arr - value_targets_arr)))
    rmse = float(math.sqrt(np.mean((value_preds_arr - value_targets_arr) ** 2)))
    if np.std(value_targets_arr) == 0 or np.std(value_preds_arr) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(value_targets_arr, value_preds_arr)[0, 1])
    sign_acc = float(np.mean(np.sign(value_targets_arr) == np.sign(value_preds_arr)))

    policy_logits_t = torch.from_numpy(policy_logits_arr)
    policy_targets_t = torch.from_numpy(policy_targets)
    policy_masks_t = torch.from_numpy(policy_masks)
    best_action_t = torch.from_numpy(best_action_ids)
    policy_loss = float(masked_policy_loss(policy_logits_t, policy_targets_t, policy_masks_t).item())
    policy_acc = float(masked_policy_accuracy(policy_logits_t, policy_masks_t, best_action_t).item())

    return {
        "value_mae": mae,
        "value_rmse": rmse,
        "value_corr": corr,
        "value_sign_acc": sign_acc,
        "policy_loss": policy_loss,
        "policy_acc": policy_acc,
        "count": float(len(features)),
    }


def evaluate_bucket_files(model: DoubleDummyMLP, dataset_files: list[Path]) -> dict[str, float]:
    """把一个桶下的多个数据文件合并评估。"""
    all_samples = []

    for dataset_file in dataset_files:
        loaded = load_bucket_dataset(dataset_file)
        all_samples.extend(loaded["samples"])

    arrays = prepare_multi_head_arrays(all_samples)
    features = arrays["features"]
    value_targets = arrays["value_targets"].squeeze(-1).astype(np.float64)
    policy_targets = arrays["policy_targets"]
    policy_masks = arrays["policy_masks"]
    best_action_ids = arrays["best_action_ids"]

    value_preds = []
    policy_logits_list = []
    for feature in features:
        outputs = model(feature)
        value_preds.append(float(outputs["value"].squeeze().item()))
        policy_logits_list.append(outputs["policy_logits"].squeeze(0).cpu().numpy())

    value_preds_arr = np.array(value_preds, dtype=np.float64)
    policy_logits_arr = np.array(policy_logits_list, dtype=np.float32)

    mae = float(np.mean(np.abs(value_preds_arr - value_targets)))
    rmse = float(math.sqrt(np.mean((value_preds_arr - value_targets) ** 2)))
    if np.std(value_targets) == 0 or np.std(value_preds_arr) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(value_targets, value_preds_arr)[0, 1])
    sign_acc = float(np.mean(np.sign(value_targets) == np.sign(value_preds_arr)))

    policy_logits_t = torch.from_numpy(policy_logits_arr)
    policy_targets_t = torch.from_numpy(policy_targets)
    policy_masks_t = torch.from_numpy(policy_masks)
    best_action_t = torch.from_numpy(best_action_ids)
    policy_loss = float(masked_policy_loss(policy_logits_t, policy_targets_t, policy_masks_t).item())
    policy_acc = float(masked_policy_accuracy(policy_logits_t, policy_masks_t, best_action_t).item())

    return {
        "value_mae": mae,
        "value_rmse": rmse,
        "value_corr": corr,
        "value_sign_acc": sign_acc,
        "policy_loss": policy_loss,
        "policy_acc": policy_acc,
        "count": float(len(all_samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./mlp_checkpoint.pth", help="模型路径")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--xs", type=int, nargs="+", default=list(SUPPORTED_BUCKETS), help="评估的剩余牌数桶")
    args = parser.parse_args()

    encoder = SpadesFeatureEncoder()
    model = DoubleDummyMLP(input_dim=encoder.total_dim, bucket_xs=tuple(args.xs))
    model.load(args.checkpoint)

    data_dir = Path(args.data_dir)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"使用桶: {tuple(args.xs)}")

    for x in args.xs:
        matched = sorted(data_dir.glob(f"{DEFAULT_OUTPUT_PREFIX}_x{x}_n*.pt"))
        if not matched:
            print(f"x={x} | 未找到数据文件")
            continue
        metrics = evaluate_bucket_files(model, matched)
        print(
            f"x={x:2d} | count={int(metrics['count'])} | "
            f"value_MAE={metrics['value_mae']:.6f} | value_RMSE={metrics['value_rmse']:.6f} | "
            f"value_corr={metrics['value_corr']:.6f} | value_sign_acc={metrics['value_sign_acc']:.6f} | "
            f"policy_loss={metrics['policy_loss']:.6f} | policy_acc={metrics['policy_acc']:.6f}"
        )


if __name__ == "__main__":
    main()
