"""新的训练入口。

默认读取 data/ 下的 x=24/28/32 桶数据，联合训练 value head 和 policy head。
value head 拟合 `value_view`，policy head 从 `action_q_values` 生成监督信号。
 
模块函数说明（输入/输出）:

- load_training_samples(data_dir: Path, xs: list[int]) -> list[dict]
    输入: data_dir: Path（数据目录），xs: 剩余牌数桶列表
    输出: 样本列表，每个样本为 data.training_data.save_bucket_dataset 中定义的样本 dict

- main() -> None
    命令行入口，基于 argparse 接收超参数并执行联合训练。
    训练过程中会把样本转换为 arrays 并在 device 上训练模型，最终保存 checkpoint 到 --save 指定路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn

from data.training_data import DEFAULT_OUTPUT_PREFIX, SUPPORTED_BUCKETS, load_bucket_dataset
from mlp.mlp_model import DoubleDummyMLP
from mlp.training_utils import masked_policy_accuracy, masked_policy_loss, prepare_multi_head_arrays
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def load_training_samples(data_dir: Path, xs: list[int]):
    """把多个桶的数据拼成一个训练集。"""
    samples = []

    for x in xs:
        matched = sorted(data_dir.glob(f"{DEFAULT_OUTPUT_PREFIX}_x{x}_n*.pt"))
        if not matched:
            raise FileNotFoundError(f"找不到 x={x} 的数据文件: {data_dir / (DEFAULT_OUTPUT_PREFIX + f'_x{x}_n*.pt')}")

        for dataset_file in matched:
            loaded = load_bucket_dataset(dataset_file)
            samples.extend(loaded["samples"])

    if not samples:
        raise FileNotFoundError(f"在 {data_dir} 中没有找到任何训练数据")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xs", type=int, nargs="+", default=list(SUPPORTED_BUCKETS), help="训练所用的剩余牌数桶")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--value_weight", type=float, default=1.0, help="value loss 权重")
    parser.add_argument("--policy_weight", type=float, default=0.0, help="policy loss 权重")
    parser.add_argument("--policy_temperature", type=float, default=0.3, help="policy 监督温度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save", type=str, default="./mlp_checkpoint.pth", help="模型保存路径")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    encoder = SpadesFeatureEncoder()
    model = DoubleDummyMLP(input_dim=encoder.total_dim, bucket_xs=tuple(args.xs))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"设备: {device}")
    print(f"使用桶: {tuple(args.xs)}")

    samples = load_training_samples(data_dir, args.xs)
    arrays = prepare_multi_head_arrays(samples, policy_temperature=args.policy_temperature)
    features = arrays["features"]
    value_targets = arrays["value_targets"]
    policy_targets = arrays["policy_targets"]
    policy_masks = arrays["policy_masks"]
    best_action_ids = arrays["best_action_ids"]
    print(f"数据量: {len(features)}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Keep full datasets on CPU and move only minibatches to `device` to avoid OOM
    X_t = torch.from_numpy(features)
    value_t = torch.from_numpy(value_targets)
    policy_target_t = torch.from_numpy(policy_targets)
    policy_mask_t = torch.from_numpy(policy_masks)
    best_action_t = torch.from_numpy(best_action_ids)

    for epoch in range(args.epochs):
        # Permute on CPU tensors
        permutation = torch.randperm(len(X_t))
        X_t = X_t[permutation]
        value_t = value_t[permutation]
        policy_target_t = policy_target_t[permutation]
        policy_mask_t = policy_mask_t[permutation]
        best_action_t = best_action_t[permutation]

        model.train()
        running_value_loss = 0.0
        running_policy_loss = 0.0
        running_total_loss = 0.0
        batches = 0
        for start in range(0, len(X_t), args.batch_size):
            end = min(start + args.batch_size, len(X_t))
            # Move minibatch to device
            batch_x = X_t[start:end].to(device)
            batch_value = value_t[start:end].to(device)
            batch_policy_target = policy_target_t[start:end].to(device)
            batch_policy_mask = policy_mask_t[start:end].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            value_pred = outputs["value"]
            policy_logits = outputs["policy_logits"]
            value_loss = criterion(value_pred, batch_value)
            policy_loss = masked_policy_loss(policy_logits, batch_policy_target, batch_policy_mask)
            loss = args.value_weight * value_loss + args.policy_weight * policy_loss
            loss.backward()
            optimizer.step()

            running_value_loss += value_loss.item()
            running_policy_loss += policy_loss.item()
            running_total_loss += loss.item()
            batches += 1

        avg_value_loss = running_value_loss / max(batches, 1)
        avg_policy_loss = running_policy_loss / max(batches, 1)
        avg_total_loss = running_total_loss / max(batches, 1)
        model.eval()
        with torch.no_grad():
            # Evaluate in minibatches to avoid moving full dataset to GPU
            eval_value_loss = 0.0
            eval_policy_loss = 0.0
            eval_policy_acc = 0.0
            eval_batches = 0
            eval_batch_size = args.batch_size
            for estart in range(0, len(X_t), eval_batch_size):
                eend = min(estart + eval_batch_size, len(X_t))
                ex = X_t[estart:eend].to(device)
                ev = value_t[estart:eend].to(device)
                ept = policy_target_t[estart:eend].to(device)
                epm = policy_mask_t[estart:eend].to(device)
                eba = best_action_t[estart:eend].to(device)
                eout = model(ex)
                eval_value_loss += criterion(eout["value"], ev).item() * (eend - estart)
                eval_policy_loss += masked_policy_loss(eout["policy_logits"], ept, epm).item() * (eend - estart)
                eval_policy_acc += masked_policy_accuracy(eout["policy_logits"], epm, eba).item() * (eend - estart)
                eval_batches += (eend - estart)
            if eval_batches > 0:
                eval_value_loss = eval_value_loss / eval_batches
                eval_policy_loss = eval_policy_loss / eval_batches
                eval_policy_acc = eval_policy_acc / eval_batches

        # free any cached GPU memory to reduce fragmentation
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_total={avg_total_loss:.6f} | train_value={avg_value_loss:.6f} | train_policy={avg_policy_loss:.6f} | "
            f"eval_value={eval_value_loss:.6f} | eval_policy={eval_policy_loss:.6f} | policy_acc={eval_policy_acc:.6f}"
        )

    model.save(args.save)
    print(f"模型已保存: {args.save}")


if __name__ == "__main__":
    main()
