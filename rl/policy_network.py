"""
Policy MLP 网络。

将 1229 维局面特征映射为 52 张牌的出牌 logits，
用于 RL policy gradient 训练。

网络结构:
- 输入: 1229 维 float32 特征向量
- 隐藏层: [512, 256] 带 ReLU
- 输出: 52 维 logits (每张牌一个)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    """策略网络: 局面特征 -> 52 张牌的 logits。"""

    def __init__(
        self,
        input_dim: int = 1229,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, 52)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """前向传播，返回 52 维 logits。

        输入:
        - x: (N, input_dim) 或 (input_dim,) 的特征

        输出:
        - logits: (N, 52) 或 (52,) 未归一化的策略 logits
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        logits = self.policy_head(self.backbone(x))
        if squeeze:
            logits = logits.squeeze(0)
        return logits

    def save(self, path: str) -> None:
        """保存策略网络权重。"""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str | torch.device = "cpu") -> None:
        """加载策略网络权重。"""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()
