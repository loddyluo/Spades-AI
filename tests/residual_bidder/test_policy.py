from __future__ import annotations

import inspect
import math
import random
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from torch import nn

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.checkpoint import CalibrationTuple
from residual_bidder.nsfp import NSFPObservation
from residual_bidder.policy import (
    ActingBidPolicy,
    BidDistribution,
    NSFPArgmaxPolicy,
    StochasticResidualPolicy,
    geometric_tail,
    stable_inverse_cdf,
)
from residual_bidder.random_tape import BidSamplingKey, policy_uniform


POLICY_ID = "a" * 64
NSFP_HASH = "b" * 64


def _observation(center: BidAction = BidAction.BID_6) -> NSFPObservation:
    scores = torch.linspace(-3.0, 3.0, 14)
    scores[int(center)] = 20.0
    return NSFPObservation(
        encoded_149=torch.linspace(-1.0, 1.0, 149),
        raw_logits_16=torch.zeros(16),
        legal_scores_14=scores,
        center=center,
    )


class _FrozenNSFPDouble:
    def __init__(self, observation: NSFPObservation) -> None:
        self.observation = observation
        self.observe_calls = 0

    def observe(self, state: object) -> NSFPObservation:
        self.observe_calls += 1
        return self.observation

    def observe_batch(self, states: Sequence[object]) -> list[NSFPObservation]:
        return [self.observation for _ in states]


class _EnsembleDouble(nn.Module):
    def __init__(self, outputs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("outputs", outputs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.outputs.to(device=values.device)


def _policy(
    *,
    outputs: torch.Tensor | None = None,
    calibration: CalibrationTuple | None = None,
    center: BidAction = BidAction.BID_6,
    checkpoint_nsfp_sha256: str = NSFP_HASH,
) -> StochasticResidualPolicy:
    if outputs is None:
        outputs = torch.tensor(
            [[1.0, 0.0], [2.0, 2.0], [3.0, 4.0], [4.0, 6.0], [5.0, 8.0]]
        )
    if calibration is None:
        calibration = CalibrationTuple(0.5, 0.75, 0.2, 0.8)
    return StochasticResidualPolicy(
        _FrozenNSFPDouble(_observation(center)),
        _EnsembleDouble(outputs),
        calibration,
        POLICY_ID,
        expected_nsfp_sha256=NSFP_HASH,
        checkpoint_nsfp_sha256=checkpoint_nsfp_sha256,
    )


def _key(index: int = 0) -> BidSamplingKey:
    return BidSamplingKey(77, f"deal-{index}", "room-A", 2, 1)


def test_geometric_tail_exact_formula_and_uniform_boundary() -> None:
    tail = geometric_tail(BidAction.BID_2, 0.5)
    expected = torch.tensor(
        [0.5**abs(action - 2) for action in range(14)], dtype=torch.float64
    )
    expected /= expected.sum()

    assert tail.dtype == torch.float64
    assert torch.equal(tail, expected)
    assert torch.equal(
        geometric_tail(BidAction.BID_9, 1.0),
        torch.full((14,), 1.0 / 14.0, dtype=torch.float64),
    )


def test_distribution_uses_divided_by_100_conservative_q_formula_exactly() -> None:
    policy = _policy()

    distribution = policy.probabilities(object(), strict=True)

    outputs = torch.tensor(
        [[1.0, 0.0], [2.0, 2.0], [3.0, 4.0], [4.0, 6.0], [5.0, 8.0]],
        dtype=torch.float64,
    )
    means = outputs.mean(dim=0)
    stds = outputs.std(dim=0, unbiased=False)
    local_values = torch.stack(
        (means[0] - 0.5 * stds[0], torch.tensor(0.0), means[1] - 0.5 * stds[1])
    )
    local = torch.softmax(local_values / 0.75, dim=0)
    scattered = torch.zeros(14, dtype=torch.float64)
    scattered[5:8] = local
    expected = 0.8 * scattered + 0.2 * geometric_tail(BidAction.BID_6, 0.8)
    expected /= expected.sum()

    assert distribution.center is BidAction.BID_6
    assert distribution.local_values == (
        float(local_values[0]),
        0.0,
        float(local_values[2]),
    )
    assert distribution.probabilities == tuple(expected.tolist())
    assert distribution.policy_id == POLICY_ID


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (torch.zeros(5, 2), BidAction.BID_6),
        (torch.tensor([[0.0, -1.0]]).repeat(5, 1), BidAction.BID_6),
        (torch.tensor([[1.0, 0.0]]).repeat(5, 1), BidAction.BID_5),
        (torch.tensor([[-1.0, 1.0]]).repeat(5, 1), BidAction.BID_7),
    ],
)
def test_zero_temperature_uses_center_lower_upper_tie_priority(
    outputs: torch.Tensor, expected: BidAction
) -> None:
    policy = _policy(
        outputs=outputs,
        calibration=CalibrationTuple(0.0, 0.0, 0.0, 1.0),
    )

    distribution = policy.probabilities(object(), strict=True)

    assert distribution.probabilities[int(expected)] == 1.0
    assert sum(distribution.probabilities) == 1.0


def test_boundary_center_masks_absent_local_value() -> None:
    distribution = _policy(center=BidAction.NIL).probabilities(object(), strict=True)

    assert distribution.local_values[0] is None
    assert distribution.local_values[1] == 0.0
    assert distribution.local_values[2] is not None


@pytest.mark.parametrize("epsilon", [0.0, 0.2, 1.0])
def test_final_distribution_is_finite_nonnegative_normalized_and_full_support_iff_tail_mixed(
    epsilon: float,
) -> None:
    distribution = _policy(
        calibration=CalibrationTuple(0.5, 0.75, epsilon, 0.8)
    ).probabilities(object(), strict=True)

    probabilities = distribution.probabilities
    assert len(probabilities) == 14
    assert all(math.isfinite(value) and value >= 0.0 for value in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=3e-16)
    assert all(value > 0.0 for value in probabilities) is (epsilon > 0.0)


def test_stable_inverse_cdf_uses_canonical_action_order_and_open_uniform() -> None:
    probabilities = [0.0] * 14
    probabilities[2] = 0.25
    probabilities[9] = 0.75

    assert stable_inverse_cdf(probabilities, math.nextafter(0.0, 1.0)) is BidAction.BID_2
    assert stable_inverse_cdf(probabilities, 0.25) is BidAction.BID_2
    assert stable_inverse_cdf(probabilities, math.nextafter(0.25, 1.0)) is BidAction.BID_9
    assert stable_inverse_cdf(probabilities, math.nextafter(1.0, 0.0)) is BidAction.BID_9
    for bad_uniform in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError):
            stable_inverse_cdf(probabilities, bad_uniform)


def test_both_concrete_policies_satisfy_acting_protocol() -> None:
    stochastic = _policy()
    legacy = NSFPArgmaxPolicy(_FrozenNSFPDouble(_observation()))

    assert isinstance(stochastic, ActingBidPolicy)
    assert isinstance(legacy, ActingBidPolicy)
    assert legacy.probabilities(object(), strict=True).probabilities[6] == 1.0


def test_probabilities_batch_has_single_state_semantics_and_exact_repeat_identity() -> None:
    policy = _policy()
    state = object()

    individual = policy.probabilities(state, strict=True)
    batched = policy.probabilities_batch([state, state], strict=True)

    assert batched == [individual, individual]
    assert batched[0].probabilities == batched[1].probabilities


def test_probabilities_and_sample_share_the_private_distribution_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    calls = 0
    original = policy._distribution_from_observation

    def counted(observation: NSFPObservation) -> BidDistribution:
        nonlocal calls
        calls += 1
        return original(observation)

    monkeypatch.setattr(policy, "_distribution_from_observation", counted)

    declared = policy.probabilities(object(), strict=True)
    decision = policy.sample(
        object(), [to_local_bid(action) for action in BidAction], _key(), strict=True
    )

    assert calls == 2
    assert decision.distribution == declared


@pytest.mark.parametrize(
    ("outputs", "calibration", "checkpoint_hash", "reason_fragment"),
    [
        (torch.zeros(4, 2), CalibrationTuple(0.0, 1.0, 0.1, 0.8), NSFP_HASH, "shape"),
        (
            torch.full((5, 2), float("nan")),
            CalibrationTuple(0.0, 1.0, 0.1, 0.8),
            NSFP_HASH,
            "finite",
        ),
        (torch.zeros(5, 2), CalibrationTuple(0.0, -1.0, 0.1, 0.8), NSFP_HASH, "calibration"),
        (torch.zeros(5, 2), CalibrationTuple(0.0, 1.0, 0.1, 0.8), "c" * 64, "hash"),
    ],
)
def test_formal_failures_raise_while_runtime_returns_structured_nsfp_fallback(
    outputs: torch.Tensor,
    calibration: CalibrationTuple,
    checkpoint_hash: str,
    reason_fragment: str,
) -> None:
    policy = _policy(
        outputs=outputs,
        calibration=calibration,
        checkpoint_nsfp_sha256=checkpoint_hash,
    )
    legal = [to_local_bid(action) for action in BidAction]

    with pytest.raises(ValueError):
        policy.sample(object(), legal, _key(), strict=True)

    fallback_distribution = policy.probabilities(object(), strict=False)
    decision = policy.sample(object(), legal, _key(), strict=False)

    assert decision.action is BidAction.BID_6
    assert fallback_distribution == decision.distribution
    assert decision.distribution.probabilities == tuple(
        1.0 if action is BidAction.BID_6 else 0.0 for action in BidAction
    )
    assert decision.effective_policy_id == "legacy-nsfp-fallback"
    assert decision.fallback_reason is not None
    assert decision.fallback_reason.startswith("residual-policy-error:")
    assert reason_fragment in decision.fallback_reason.lower()


def test_missing_selected_legal_string_is_strict_drift_and_runtime_fallback() -> None:
    outputs = torch.tensor([[-1.0, 3.0]]).repeat(5, 1)
    policy = _policy(
        outputs=outputs,
        calibration=CalibrationTuple(0.0, 0.0, 0.0, 1.0),
    )
    legal_without_upper = [to_local_bid(action) for action in BidAction if action != 7]

    with pytest.raises(ValueError, match="legal bid"):
        policy.sample(object(), legal_without_upper, _key(), strict=True)

    decision = policy.sample(object(), legal_without_upper, _key(), strict=False)
    assert decision.action is BidAction.BID_6
    assert decision.effective_policy_id == "legacy-nsfp-fallback"
    assert decision.fallback_reason is not None and "legal-action-drift" in decision.fallback_reason


def test_no_decision_returns_when_its_expected_canonical_string_is_absent() -> None:
    policy = _policy(
        outputs=torch.zeros(5, 2),
        calibration=CalibrationTuple(0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="fallback legal bid"):
        policy.sample(object(), ["bid_5", "bid_7"], _key(), strict=False)


def test_sampling_calls_no_mutable_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("sampling must use only the supplied policy tape")

    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(np.random, "random", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    legal = [to_local_bid(action) for action in BidAction]

    first = _policy().sample(object(), legal, _key(91), strict=True)
    second = _policy().sample(object(), legal, _key(91), strict=True)

    assert first == second
    assert first.uniform == policy_uniform(_key(91))


def test_fixed_policy_tape_empirically_matches_declared_distribution() -> None:
    policy = _policy(calibration=CalibrationTuple(0.5, 0.75, 0.35, 0.9))
    probabilities = policy.probabilities(object(), strict=True).probabilities
    sample_size = 200_000
    counts = [0] * 14

    for index in range(sample_size):
        uniform = policy_uniform(_key(index))
        counts[int(stable_inverse_cdf(probabilities, uniform))] += 1

    frequencies = [count / sample_size for count in counts]
    tolerances = [
        6.0 * math.sqrt(probability * (1.0 - probability) / sample_size) + 1.0 / sample_size
        for probability in probabilities
    ]
    assert all(
        abs(actual - expected) <= tolerance
        for actual, expected, tolerance in zip(
            frequencies, probabilities, tolerances, strict=True
        )
    )


def test_public_shapes_and_constructor_do_not_accept_rng_or_shuffle_seed() -> None:
    assert len(BidDistribution(tuple([1.0] + [0.0] * 13), BidAction.NIL, (None, 0.0, 0.0), POLICY_ID).probabilities) == 14
    constructor = inspect.signature(StochasticResidualPolicy)
    assert "rng" not in constructor.parameters
    assert "shuffle_seed" not in constructor.parameters
