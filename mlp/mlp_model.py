"""面向训练数据桶的双头 MLP 模型。

说明：
- value head 负责回归 `value_view`。
- policy head 负责给 52 张牌输出动作偏好。
- `action_q_values` 会在训练脚本里转成 policy 监督信号。
- 模型本身是桶无关的，但保留 `bucket_xs` 作为接口，便于训练脚本按桶组织数据。
 
模块函数说明（输入/输出）：

- DoubleDummyMLP(...)
    输入: input_dim (int), hidden_dims (list[int]|None), value_output_dim (int), policy_output_dim (int), bucket_xs (tuple)
    输出: nn.Module 实例，forward 返回 dict{"value": Tensor (N,1), "policy_logits": Tensor (N,policy_dim)}

- forward(x) -> dict
    输入: x: torch.Tensor or np.ndarray, shape (N, input_dim) 或 (input_dim,)
    输出: dict, 包含 "value" 和 "policy_logits" 两个 torch.Tensor

- predict(features) -> float 或 np.ndarray
    输入: features: np.ndarray 或 torch.Tensor
    输出: 单个样本时返回 float, 批量时返回 np.ndarray

- predict_policy_logits(features) -> np.ndarray
    输入: features
    输出: numpy 数组或 logits

- save(path) / load(path, device="cpu")
    输入/输出: 保存/加载模型参数到文件；load 可指定加载到的设备
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DoubleDummyMLP(nn.Module):
    """用于拟合当前行动方价值和动作偏好的双头网络。"""

    def __init__(
        self,
        input_dim: int = 1229,
        hidden_dims: list[int] | None = None,
        value_output_dim: int = 1,
        policy_output_dim: int = 52,
        bucket_xs: tuple[int, ...] = (24, 28, 32),
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4096,2048,1024, 512, 256]

        self.input_dim = input_dim
        self.bucket_xs = tuple(bucket_xs)

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, value_output_dim)
        self.policy_head = nn.Linear(prev, policy_output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)
        hidden = self.backbone(x)
        value = self.value_head(hidden)
        policy_logits = self.policy_head(hidden)
        return {
            "value": value,
            "policy_logits": policy_logits,
        }

    def predict(self, features: np.ndarray | torch.Tensor):
        """只返回 value head 的预测值。"""
        self.eval()
        squeeze_out = isinstance(features, np.ndarray) and features.ndim == 1
        with torch.no_grad():
            pred = self.forward(features)["value"]
        if squeeze_out:
            return pred.squeeze().item()
        return pred.squeeze(-1).cpu().numpy()

    def predict_policy_logits(self, features: np.ndarray | torch.Tensor):
        """返回 policy head 的原始 logits。"""
        self.eval()
        with torch.no_grad():
            pred = self.forward(features)["policy_logits"]
        return pred.squeeze(0).cpu().numpy() if pred.shape[0] == 1 else pred.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str | torch.device = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()
