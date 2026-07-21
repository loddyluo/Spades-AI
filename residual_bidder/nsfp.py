"""Frozen access to the collaborator NSFP bidder's public observation path."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from spades_ai.models.bid_encoder import BidEncoder
from spades_ai.models.bid_mlp import BidMLP
from trick_taking.game_state import GameState

from residual_bidder.actions import BidAction, choose_center, legal_scores_14


def _load_reference_bridge() -> ModuleType:
    bridge_path = Path(__file__).resolve().parents[1] / "evaluate" / "GO-MCTS" / "bridge.py"
    spec = importlib.util.spec_from_file_location("_residual_bidder_reference_bridge", bridge_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference bridge at {bridge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REFERENCE_BRIDGE = _load_reference_bridge()


@dataclass(frozen=True)
class NSFPObservation:
    """Only the frozen public input, final logits, normalized scores, and action."""

    encoded_149: torch.Tensor
    raw_logits_16: torch.Tensor
    legal_scores_14: torch.Tensor
    center: BidAction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_float_tensor(
    tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != len(expected_shape) or tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


class FrozenNSFP:
    """A hash-pinned, inference-only wrapper around the existing NSFP bridge."""

    def __init__(self, model: BidMLP, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._encoder = BidEncoder()

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        expected_sha256: str,
        device: torch.device,
    ) -> FrozenNSFP:
        checkpoint = Path(checkpoint)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("expected_sha256 must be a 64-character SHA-256 digest")
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != expected_sha256.lower():
            raise ValueError(
                f"checkpoint SHA-256 mismatch for {checkpoint}: "
                f"expected {expected_sha256.lower()}, got {actual_sha256}"
            )

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint must deserialize to a state dictionary")
        model = BidMLP()
        model.load_state_dict(state_dict, strict=True)
        for name, tensor in model.state_dict().items():
            if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"checkpoint tensor {name!r} must be finite floating point")
        model.to(device)
        model.requires_grad_(False)
        model.eval()
        return cls(model=model, device=device)

    def _encode(self, state: GameState) -> torch.Tensor:
        go_state = _REFERENCE_BRIDGE.to_go_state(state)
        encoded = self._encoder.encode(
            list(go_state.hands[go_state.current_player]),
            list(go_state.bids),
            len(go_state.bids),
        )
        _validate_float_tensor(encoded, (149,), "encoded_149")
        return encoded.cpu()

    def _observe_encoded(self, encoded: torch.Tensor) -> NSFPObservation:
        with torch.inference_mode():
            raw = self.model(encoded.unsqueeze(0).to(self.device))
        _validate_float_tensor(raw, (1, 16), "model output")
        raw_logits = raw.squeeze(0).cpu()
        scores = legal_scores_14(raw_logits)
        return NSFPObservation(
            encoded_149=encoded,
            raw_logits_16=raw_logits,
            legal_scores_14=scores,
            center=choose_center(scores),
        )

    def observe(self, state: GameState) -> NSFPObservation:
        return self._observe_encoded(self._encode(state))

    def observe_batch(self, states: Sequence[GameState]) -> list[NSFPObservation]:
        if not isinstance(states, Sequence):
            raise TypeError("states must be a sequence")
        return [self._observe_encoded(self._encode(state)) for state in states]
