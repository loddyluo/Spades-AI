"""Leakage-safe residual features and an independently seeded Q ensemble."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from residual_bidder.actions import LocalNeighborhood, neighborhood
from residual_bidder.nsfp import NSFPObservation


INPUT_DIM = 167
OUTPUT_DIM = 2
ENSEMBLE_MEMBERS = 5
DEFAULT_MARGIN_DIVISOR = 13.47


@dataclass(frozen=True)
class ResidualInput:
    """The complete public input available to the residual bidder."""

    values: torch.Tensor
    neighborhood: LocalNeighborhood
    alternative_mask: torch.Tensor


def _validate_public_observation(obs: NSFPObservation) -> None:
    if not isinstance(obs, NSFPObservation):
        raise TypeError("obs must be an NSFPObservation")
    for name, tensor, shape in (
        ("encoded_149", obs.encoded_149, (149,)),
        ("legal_scores_14", obs.legal_scores_14, (14,)),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"obs.{name} must be a torch.Tensor")
        if tensor.shape != shape:
            raise ValueError(f"obs.{name} must have shape {shape}")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"obs.{name} must have a floating-point dtype")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"obs.{name} must contain only finite values")
    if obs.encoded_149.device != obs.legal_scores_14.device:
        raise ValueError("public observation tensors must be on the same device")
    if obs.encoded_149.dtype != obs.legal_scores_14.dtype:
        raise ValueError("public observation tensors must have the same dtype")
    if obs.center != neighborhood(obs.center).center:
        raise TypeError("obs.center must be a canonical BidAction")


def build_residual_input(
    obs: NSFPObservation, margin_divisor: float = DEFAULT_MARGIN_DIVISOR
) -> ResidualInput:
    """Build the fixed 167-vector solely from a frozen public NSFP observation."""

    _validate_public_observation(obs)
    if not isinstance(margin_divisor, (int, float)) or not math.isfinite(
        float(margin_divisor)
    ) or margin_divisor <= 0:
        raise ValueError("margin_divisor must be finite and positive")

    local = neighborhood(obs.center)
    features = torch.zeros(
        INPUT_DIM,
        dtype=obs.encoded_149.dtype,
        device=obs.encoded_149.device,
    )
    features[:149] = obs.encoded_149
    center_index = int(obs.center)
    features[149 + center_index] = 1.0
    center_score = obs.legal_scores_14[center_index]

    for slot, alternative in enumerate((local.lower, local.upper)):
        if alternative is not None:
            features[163 + slot] = (
                center_score - obs.legal_scores_14[int(alternative)]
            ) / float(margin_divisor)
            features[165 + slot] = 1.0

    return ResidualInput(
        values=features,
        neighborhood=local,
        alternative_mask=features[165:167],
    )


class ResidualBlock(nn.Module):
    """A width-preserving residual block with post-add activation."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.linear_in = nn.Linear(width, width)
        self.normalization = nn.LayerNorm(width)
        self.activation_in = nn.SiLU()
        self.linear_out = nn.Linear(width, width)
        self.activation_out = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.linear_out(self.activation_in(self.normalization(self.linear_in(x))))
        return self.activation_out(x + residual)


class ResidualQMember(nn.Module):
    """One 167-to-2 lower/upper residual-Q network."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(INPUT_DIM, 256)
        self.input_normalization = nn.LayerNorm(256)
        self.input_activation = nn.SiLU()
        self.residual_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.bottleneck = nn.Linear(256, 128)
        self.bottleneck_normalization = nn.LayerNorm(128)
        self.bottleneck_activation = nn.SiLU()
        self.output_layer = nn.Linear(128, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1 or x.shape[-1] != INPUT_DIM:
            raise ValueError(f"x must have trailing dimension {INPUT_DIM}")
        hidden = self.input_activation(self.input_normalization(self.input_layer(x)))
        hidden = self.residual_blocks(hidden)
        hidden = self.bottleneck_activation(
            self.bottleneck_normalization(self.bottleneck(hidden))
        )
        return self.output_layer(hidden)


class ResidualQEnsemble(nn.Module):
    """Five independent Q members, never a selector over five policies."""

    def __init__(self, member_init_seeds: tuple[int, int, int, int, int]) -> None:
        super().__init__()
        if (
            not isinstance(member_init_seeds, tuple)
            or len(member_init_seeds) != ENSEMBLE_MEMBERS
            or any(type(seed) is not int for seed in member_init_seeds)
            or len(set(member_init_seeds)) != ENSEMBLE_MEMBERS
        ):
            raise ValueError("member_init_seeds must contain five distinct integers")
        self.member_init_seeds = member_init_seeds
        members: list[ResidualQMember] = []
        for seed in member_init_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                members.append(ResidualQMember())
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(x) for member in self.members], dim=0)

    def mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self(x)
        return outputs.mean(dim=0), outputs.std(dim=0, unbiased=False)


def _validate_alternative_mask(values: torch.Tensor, alternative_mask: torch.Tensor) -> None:
    if not isinstance(values, torch.Tensor) or not isinstance(alternative_mask, torch.Tensor):
        raise TypeError("values and alternative_mask must be torch tensors")
    if values.shape != alternative_mask.shape or values.shape[-1:] != (OUTPUT_DIM,):
        raise ValueError("values and alternative_mask must have equal shapes ending in 2")
    if not bool(torch.isfinite(alternative_mask).all().item()):
        raise ValueError("alternative_mask must be finite")
    if not bool(((alternative_mask == 0) | (alternative_mask == 1)).all().item()):
        raise ValueError("alternative_mask must contain only zero or one")


def mask_invalid_alternatives(
    values: torch.Tensor,
    alternative_mask: torch.Tensor,
    invalid_value: float = -math.inf,
) -> torch.Tensor:
    """Mask missing lower/upper alternatives in their fixed output slots."""

    _validate_alternative_mask(values, alternative_mask)
    return values.masked_fill(alternative_mask.to(dtype=torch.bool).logical_not(), invalid_value)


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alternative_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over legal alternatives without slot remapping."""

    if not isinstance(targets, torch.Tensor) or targets.shape != predictions.shape:
        raise ValueError("targets must be a tensor with the predictions shape")
    _validate_alternative_mask(predictions, alternative_mask)
    selected = (predictions - targets).square().masked_select(
        alternative_mask.to(dtype=torch.bool)
    )
    if selected.numel() == 0:
        raise ValueError("at least one alternative must be legal")
    return selected.mean()
