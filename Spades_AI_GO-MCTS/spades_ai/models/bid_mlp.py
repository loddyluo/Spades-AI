"""BidMLP: 149 -> 16 bid-logit predictor.

Architecture:
    Linear(149, 256) -> ReLU -> Dropout(0.1)
    Linear(256, 256) -> ReLU -> Dropout(0.1)
    Linear(256, 128) -> ReLU
    Linear(128,  16)

The 16 output logits correspond to the same bid-index mapping as BidEncoder:
  0-13  normal bids, 14  Nil, 15  Blind Nil.
"""
from __future__ import annotations

import torch.nn as nn


class BidMLP(nn.Module):
    """Multi-layer perceptron for bid prediction from encoded hand features."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(149, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: float tensor of shape (B, 149).

        Returns:
            logits of shape (B, 16).
        """
        return self.net(x)
