from __future__ import annotations

import pytest
import torch

from residual_bidder.actions import (
    BidAction,
    LocalNeighborhood,
    choose_center,
    from_local_bid,
    legal_scores_14,
    neighborhood,
    to_local_bid,
)


def test_legal_scores_collapse_raw_aliases_into_fourteen_actions() -> None:
    raw = torch.arange(16, dtype=torch.float32)
    raw[0], raw[1] = 9.0, 3.0
    raw[14], raw[15] = 4.0, 8.0

    scores = legal_scores_14(raw)

    assert scores.shape == (14,)
    assert scores.dtype == raw.dtype
    assert scores.tolist() == [8.0, 9.0, *map(float, range(2, 14))]


def test_choose_center_uses_lowest_canonical_index_for_ties() -> None:
    scores = torch.full((14,), -1.0)
    scores[0] = scores[1] = scores[13] = 5.0

    assert choose_center(scores) is BidAction.NIL


@pytest.mark.parametrize(
    ("center", "expected"),
    [
        (BidAction.NIL, LocalNeighborhood(BidAction.NIL, None, BidAction.BID_1)),
        (BidAction.BID_1, LocalNeighborhood(BidAction.BID_1, BidAction.NIL, BidAction.BID_2)),
        (BidAction.BID_7, LocalNeighborhood(BidAction.BID_7, BidAction.BID_6, BidAction.BID_8)),
        (BidAction.BID_13, LocalNeighborhood(BidAction.BID_13, BidAction.BID_12, None)),
    ],
)
def test_neighborhood_has_exact_canonical_boundaries(
    center: BidAction, expected: LocalNeighborhood
) -> None:
    assert neighborhood(center) == expected


@pytest.mark.parametrize("action", list(BidAction))
def test_local_bid_conversion_round_trips_canonical_actions(action: BidAction) -> None:
    expected = "nil" if action is BidAction.NIL else f"bid_{int(action)}"

    assert to_local_bid(action) == expected
    assert from_local_bid(expected) is action


def test_local_bid_aliases_collapse_to_their_canonical_actions() -> None:
    assert from_local_bid("blind_nil") is BidAction.NIL
    assert from_local_bid("bid_0") is BidAction.BID_1


@pytest.mark.parametrize(
    "raw",
    [
        torch.zeros(1, 16),
        torch.zeros(15),
        torch.zeros(16, dtype=torch.int64),
        torch.tensor([0.0] * 15 + [float("nan")]),
    ],
)
def test_legal_scores_reject_invalid_logits(raw: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        legal_scores_14(raw)


@pytest.mark.parametrize(
    "scores",
    [
        torch.zeros(1, 14),
        torch.zeros(13),
        torch.zeros(14, dtype=torch.int64),
        torch.tensor([0.0] * 13 + [float("inf")]),
    ],
)
def test_choose_center_rejects_invalid_scores(scores: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        choose_center(scores)


def test_action_helpers_reject_noncanonical_values() -> None:
    with pytest.raises((TypeError, ValueError)):
        neighborhood(14)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        to_local_bid(14)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        from_local_bid("bid_14")
    with pytest.raises(ValueError):
        from_local_bid("bid_01")
