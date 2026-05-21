"""Value-only MLP for full-information x=24 evaluation.

This module defines a small value network that predicts a scaled value_view
(`value_view / value_scale`) from full-information features.

Module contents (inputs/outputs):
- FullInfoValueMLP(...)
    Input: input_dim (int), hidden_dims (list[int]|None), value_output_dim (int)
    Output: nn.Module instance; forward returns Tensor (N,1).
- FullInfoValueMLP.forward(x)
    Input: x (np.ndarray or torch.Tensor), shape (N,input_dim) or (input_dim,)
    Output: dict with key "value" -> torch.Tensor shape (N,1).
- FullInfoValueMLP.predict(features)
    Input: np.ndarray or torch.Tensor; single or batch.
    Output: float for single input or np.ndarray for batch; values are scaled.
- FullInfoValueMLP.save(path) / FullInfoValueMLP.load(path, device)
    Input: filesystem path; device string for load.
    Output: save writes weights; load returns None and updates the module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FullInfoValueMLP(nn.Module):
    """Value-only MLP for full-information features.

    The network predicts a single scalar value per state. The output is
    expected to be trained against `value_view / value_scale`.
    """

    def __init__(
        self,
        input_dim: int = 1385,
        hidden_dims: list[int] | None = None,
        value_output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.input_dim = input_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, value_output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
        """Forward pass for value prediction.

        Input:
        - x: torch.Tensor or np.ndarray, shape (N,input_dim) or (input_dim,)

        Output:
        - dict with key "value" -> torch.Tensor shape (N,1)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)
        hidden = self.backbone(x)
        value = self.value_head(hidden)
        return {"value": value}

    def predict(self, features: np.ndarray | torch.Tensor):
        """Return scaled value predictions.

        Input:
        - features: np.ndarray or torch.Tensor; single or batch.

        Output:
        - float (single) or np.ndarray (batch) of scaled values.
        """
        self.eval()
        squeeze_out = isinstance(features, np.ndarray) and features.ndim == 1
        with torch.no_grad():
            pred = self.forward(features)["value"]
        if squeeze_out:
            return pred.squeeze().item()
        return pred.squeeze(-1).cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights.

        Input:
        - path: filesystem path.

        Output:
        - None. Writes weights to disk.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str | torch.device = "cpu") -> None:
        """Load model weights and move to device.

        Input:
        - path: filesystem path.
        - device: torch device or string.

        Output:
        - None. Updates module parameters in-place.
        """
        state_dict = torch.load(path, map_location=device)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            inferred_hidden_dims, use_dropout = self._infer_architecture_from_state_dict(state_dict)
            self._rebuild_architecture(inferred_hidden_dims, use_dropout=use_dropout)
            self.load_state_dict(state_dict)
        self.to(device)
        self.eval()

    def _rebuild_architecture(self, hidden_dims: list[int], use_dropout: bool | None = None) -> None:
        """Rebuild the network to match a loaded checkpoint architecture."""
        layers: list[nn.Module] = []
        prev = self.input_dim
        if use_dropout is None:
            use_dropout = self.dropout > 0.0
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout and self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, 1)

    def _infer_architecture_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[int], bool]:
        """Infer hidden layer widths and dropout presence from a checkpoint state_dict."""
        linear_keys: list[tuple[int, torch.Tensor]] = []
        for key, tensor in state_dict.items():
            if not key.startswith("backbone.") or not key.endswith(".weight"):
                continue
            parts = key.split(".")
            if len(parts) != 3:
                continue
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            if tensor.ndim != 2:
                continue
            linear_keys.append((layer_idx, tensor))

        linear_keys.sort(key=lambda item: item[0])
        hidden_dims = [int(tensor.shape[0]) for _, tensor in linear_keys]
        if not hidden_dims:
            raise RuntimeError("unable to infer hidden_dims from checkpoint state_dict")
        use_dropout = False
        if len(linear_keys) >= 2:
            first_idx = linear_keys[0][0]
            second_idx = linear_keys[1][0]
            use_dropout = (second_idx - first_idx) >= 3
        return hidden_dims, use_dropout
