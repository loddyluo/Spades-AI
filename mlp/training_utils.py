"""MLP 训练与评估的公共工具。

模块内函数说明（在此简单列出每个函数的输入/输出格式，方便代码审查与自动文档）：

- _to_numpy(array_like) -> np.ndarray
    输入: torch.Tensor | np.ndarray | list; 输出: np.ndarray

- build_policy_target(action_ids, action_q_values, policy_dim=52, temperature=1.0) -> (policy_target, policy_mask)
    输入:
        - action_ids: 可迭代的动作 id 列表（card_id 范围 0..51）或 torch.Tensor
        - action_q_values: 对应的 Q 值数组或 tensor
        - policy_dim: 整数，policy 向量维度（默认 52）
        - temperature: float，用于 softmax 缩放
    输出:
        - policy_target: np.ndarray, shape=(policy_dim,), 在合法动作位置为概率分布，其余为0
        - policy_mask: np.ndarray, shape=(policy_dim,), 合法动作位置为1，其余为0

- prepare_multi_head_arrays(samples, *, policy_dim=52, value_scale=25.0, policy_temperature=1.0) -> dict
    输入:
        - samples: list[dict]，每个样本遵循 data.save_bucket_dataset 的样本格式（含 feature, value_view, action_ids, action_q_values, best_action_id）
        - policy_dim: int
        - value_scale: float，用于将原始 value_view 缩放到网络训练目标
        - policy_temperature: float
    输出: dict 包含 keys: features (N, D), value_targets (N,1), policy_targets (N,policy_dim), policy_masks (N,policy_dim), best_action_ids (N,)

- masked_policy_loss(policy_logits, policy_targets, policy_masks) -> torch.Tensor
    输入: 三个 torch.Tensor，shape 分别为 (N,policy_dim), (N,policy_dim), (N,policy_dim)
    输出: 标量 tensor，代表 batch 上的平均交叉熵损失，仅在 policy_masks=1 的位置计算

- masked_policy_accuracy(policy_logits, policy_masks, best_action_ids) -> torch.Tensor
    输入: policy_logits (N,policy_dim), policy_masks (N,policy_dim), best_action_ids (N,)
    输出: 标量 tensor，表示预测的 top-1 在 best_action_ids 上的命中率（忽略 best_action_id<0 的样本）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy(array_like: Any) -> np.ndarray:
    """把 torch / numpy / list 统一转成 numpy 数组。"""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def build_policy_target(
    action_ids: Any,
    action_q_values: Any,
    policy_dim: int = 52,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """把可行动作的 Q 值转成 policy 监督信号。

    返回：
    - policy_target: 只在合法动作位置有概率，其余位置为 0
    - policy_mask: 合法动作位置为 1，其余位置为 0
    """
    action_ids_np = _to_numpy(action_ids).astype(np.int64, copy=False)
    action_q_values_np = _to_numpy(action_q_values).astype(np.float32, copy=False)

    policy_target = np.zeros(policy_dim, dtype=np.float32)
    policy_mask = np.zeros(policy_dim, dtype=np.float32)

    if action_ids_np.size == 0:
        return policy_target, policy_mask

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    scaled_q_values = action_q_values_np / 25.0
    scaled_q_values = scaled_q_values / temperature
    scaled_q_values = scaled_q_values - float(np.max(scaled_q_values))
    exp_values = np.exp(scaled_q_values)
    exp_sum = float(np.sum(exp_values))
    if exp_sum <= 0:
        probabilities = np.full(len(action_ids_np), 1.0 / len(action_ids_np), dtype=np.float32)
    else:
        probabilities = (exp_values / exp_sum).astype(np.float32)

    policy_target[action_ids_np] = probabilities
    policy_mask[action_ids_np] = 1.0
    return policy_target, policy_mask


def prepare_multi_head_arrays(
    samples: list[dict[str, Any]],
    *,
    policy_dim: int = 52,
    value_scale: float = 25.0,
    policy_temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """把样本列表转换成训练/评估可直接使用的数组。"""
    if not samples:
        raise ValueError("samples 不能为空")

    feature_rows = []
    value_targets = []
    policy_targets = []
    policy_masks = []
    best_action_ids = []

    for sample in samples:
        feature_rows.append(_to_numpy(sample["feature"]).astype(np.float32, copy=False))
        value_targets.append([float(_to_numpy(sample["value_view"])) / value_scale])

        policy_target, policy_mask = build_policy_target(
            sample["action_ids"],
            sample["action_q_values"],
            policy_dim=policy_dim,
            temperature=policy_temperature,
        )
        policy_targets.append(policy_target)
        policy_masks.append(policy_mask)
        best_action_ids.append(int(sample["best_action_id"]))

    return {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "value_targets": np.asarray(value_targets, dtype=np.float32),
        "policy_targets": np.asarray(policy_targets, dtype=np.float32),
        "policy_masks": np.asarray(policy_masks, dtype=np.float32),
        "best_action_ids": np.asarray(best_action_ids, dtype=np.int64),
    }


def masked_policy_loss(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    policy_masks: torch.Tensor,
) -> torch.Tensor:
    """只在合法动作上计算 policy loss。"""
    if policy_logits.ndim != 2:
        raise ValueError("policy_logits 必须是二维张量")

    masked_logits = policy_logits.masked_fill(policy_masks <= 0, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    per_sample_loss = -(policy_targets * log_probs).sum(dim=1)

    valid_rows = policy_masks.sum(dim=1) > 0
    if not torch.any(valid_rows):
        return per_sample_loss.mean() * 0.0

    return per_sample_loss[valid_rows].mean()


def masked_policy_accuracy(
    policy_logits: torch.Tensor,
    policy_masks: torch.Tensor,
    best_action_ids: torch.Tensor,
) -> torch.Tensor:
    """计算 policy 头对最佳动作的命中率。"""
    masked_logits = policy_logits.masked_fill(policy_masks <= 0, -1e9)
    predicted_actions = torch.argmax(masked_logits, dim=1)
    valid_rows = best_action_ids >= 0
    if not torch.any(valid_rows):
        return predicted_actions.float().mean() * 0.0
    return (predicted_actions[valid_rows] == best_action_ids[valid_rows]).float().mean()
