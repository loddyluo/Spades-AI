"""Canonical deployed bidding actions and NSFP-logit normalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class BidAction(IntEnum):
    """The fourteen actions exposed by the deployed bidder."""

    NIL = 0
    BID_1 = 1
    BID_2 = 2
    BID_3 = 3
    BID_4 = 4
    BID_5 = 5
    BID_6 = 6
    BID_7 = 7
    BID_8 = 8
    BID_9 = 9
    BID_10 = 10
    BID_11 = 11
    BID_12 = 12
    BID_13 = 13


@dataclass(frozen=True)
class LocalNeighborhood:
    """The canonical center action and its immediate ordered neighbors."""

    center: BidAction
    lower: BidAction | None
    upper: BidAction | None


def _validate_vector(tensor: torch.Tensor, length: int, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1 or tensor.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def legal_scores_14(raw_logits_16: torch.Tensor) -> torch.Tensor:
    """Collapse the NSFP model's two pairs of aliases into 14 legal scores."""

    _validate_vector(raw_logits_16, 16, "raw_logits_16")
    nil = torch.maximum(raw_logits_16[14], raw_logits_16[15])
    bid_one = torch.maximum(raw_logits_16[0], raw_logits_16[1])
    return torch.cat((nil.unsqueeze(0), bid_one.unsqueeze(0), raw_logits_16[2:14]))


def choose_center(scores_14: torch.Tensor) -> BidAction:
    """Choose the highest-scoring action, resolving ties by canonical index."""

    _validate_vector(scores_14, 14, "scores_14")
    return BidAction(int(torch.argmax(scores_14).item()))


def _require_action(action: BidAction) -> BidAction:
    if not isinstance(action, BidAction):
        raise TypeError("action must be a BidAction")
    return action


def neighborhood(center: BidAction) -> LocalNeighborhood:
    """Return the immediate canonical neighborhood around ``center``."""

    center = _require_action(center)
    index = int(center)
    lower = BidAction(index - 1) if index > 0 else None
    upper = BidAction(index + 1) if index < int(BidAction.BID_13) else None
    return LocalNeighborhood(center=center, lower=lower, upper=upper)


def to_local_bid(action: BidAction) -> str:
    """Convert a canonical action to the local runner's bid spelling."""

    action = _require_action(action)
    return "nil" if action is BidAction.NIL else f"bid_{int(action)}"


def from_local_bid(value: str) -> BidAction:
    """Convert a formal deployable local bid string to a canonical action."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if value == "nil":
        return BidAction.NIL
    canonical_bids = {f"bid_{int(action)}": action for action in list(BidAction)[1:]}
    if value in canonical_bids:
        return canonical_bids[value]
    raise ValueError(f"unsupported local bid: {value!r}")
