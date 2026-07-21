from __future__ import annotations

import inspect
import math
import random
from collections.abc import Sequence
from dataclasses import replace
from types import MethodType

import numpy as np
import pytest
import torch
from torch import nn

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.checkpoint import (
    BidderCheckpointMeta,
    CalibrationTuple,
    build_candidate_meta,
    promote_meta,
)
from residual_bidder.model import ResidualQEnsemble
from residual_bidder.nsfp import FrozenNSFP, NSFPObservation
from residual_bidder.policy import (
    ActingBidPolicy,
    BidDistribution,
    NSFPArgmaxPolicy,
    StochasticResidualPolicy,
    geometric_tail,
    stable_inverse_cdf,
)
from residual_bidder.random_tape import BidSamplingKey, policy_uniform


NSFP_HASH = "b" * 64
SEEDS = (101, 202, 303, 404, 505)
PROVENANCE = {
    "nsfp_sha256": NSFP_HASH,
    "play_pipeline_sha256": "c" * 64,
    "config_sha256": "d" * 64,
    "dataset_manifest_sha256": "e" * 64,
}
EMPIRICAL_PROBABILITIES = (
    0.01876489258273326,
    0.020849880647481396,
    0.023166534052757107,
    0.02574059339195234,
    0.02860065932439149,
    0.2892271984308223,
    0.04741458768105883,
    0.4122246905650278,
    0.02860065932439149,
    0.02574059339195234,
    0.023166534052757107,
    0.020849880647481396,
    0.01876489258273326,
    0.01688840332445993,
)
EMPIRICAL_TOLERANCES = (
    0.0018255215873580059,
    0.0019219583621988485,
    0.002023259704669624,
    0.002129627671824137,
    0.0022412645393397185,
    0.00608804764911162,
    0.0028563084750859425,
    0.0066090282487014755,
    0.0022412645393397185,
    0.002129627671824137,
    0.002023259704669624,
    0.0019219583621988485,
    0.0018255215873580059,
    0.0017337490646041878,
)


def _observation(center: BidAction = BidAction.BID_6) -> NSFPObservation:
    scores = torch.linspace(-3.0, 3.0, 14)
    scores[int(center)] = 20.0
    return NSFPObservation(
        encoded_149=torch.linspace(-1.0, 1.0, 149),
        raw_logits_16=torch.zeros(16),
        legal_scores_14=scores,
        center=center,
    )


class _FrozenNSFPDouble(FrozenNSFP):
    def __init__(
        self,
        observation: NSFPObservation,
        checkpoint_sha256: str = NSFP_HASH,
    ) -> None:
        super().__init__(nn.Identity(), torch.device("cpu"), checkpoint_sha256)
        self.observation = observation
        self.observe_calls = 0

    def observe(self, state: object) -> NSFPObservation:
        self.observe_calls += 1
        return self.observation

    def observe_batch(self, states: Sequence[object]) -> list[NSFPObservation]:
        return [self.observation for _ in states]


_SHARED_ENSEMBLE = ResidualQEnsemble(SEEDS)
_SHARED_CANDIDATE = build_candidate_meta(
    _SHARED_ENSEMBLE,
    iteration=7,
    member_init_seeds=SEEDS,
    **PROVENANCE,
)


def _set_outputs(ensemble: ResidualQEnsemble, outputs: torch.Tensor) -> None:
    ensemble._test_outputs = outputs  # type: ignore[attr-defined]

    def fixed_forward(self: ResidualQEnsemble, values: torch.Tensor) -> torch.Tensor:
        return self._test_outputs.to(device=values.device)  # type: ignore[attr-defined]

    ensemble.forward = MethodType(fixed_forward, ensemble)  # type: ignore[method-assign]


def _promoted(
    calibration: CalibrationTuple,
    ensemble: ResidualQEnsemble = _SHARED_ENSEMBLE,
) -> BidderCheckpointMeta:
    candidate = (
        _SHARED_CANDIDATE
        if ensemble is _SHARED_ENSEMBLE
        else build_candidate_meta(
            ensemble,
            iteration=7,
            member_init_seeds=ensemble.member_init_seeds,
            **PROVENANCE,
        )
    )
    return promote_meta(candidate, calibration)


def _policy(
    *,
    outputs: torch.Tensor | None = None,
    calibration: CalibrationTuple | None = None,
    center: BidAction = BidAction.BID_6,
    checkpoint_nsfp_sha256: str = NSFP_HASH,
    ensemble: ResidualQEnsemble = _SHARED_ENSEMBLE,
    metadata: BidderCheckpointMeta | None = None,
) -> StochasticResidualPolicy:
    if outputs is None:
        outputs = torch.tensor(
            [[1.0, 0.0], [2.0, 2.0], [3.0, 4.0], [4.0, 6.0], [5.0, 8.0]]
        )
    if calibration is None:
        calibration = CalibrationTuple(0.5, 0.75, 0.2, 0.8)
    _set_outputs(ensemble, outputs)
    if metadata is None:
        metadata = _promoted(calibration, ensemble)
    return StochasticResidualPolicy(
        _FrozenNSFPDouble(_observation(center), checkpoint_nsfp_sha256),
        ensemble,
        metadata,
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


def test_geometric_tail_keeps_full_support_for_subnormal_positive_rho() -> None:
    rho = math.nextafter(0.0, 1.0)

    tail = geometric_tail(BidAction.NIL, rho)

    assert bool(torch.isfinite(tail).all().item())
    assert bool((tail > 0.0).all().item())
    assert float(tail.sum(dtype=torch.float64).item()) == 1.0


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
    assert distribution.probabilities == pytest.approx(
        tuple(expected.tolist()), rel=2e-15, abs=0.0
    )
    assert distribution.policy_id == policy.policy_id


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


def test_bid_13_boundary_masks_absent_upper_local_value() -> None:
    distribution = _policy(center=BidAction.BID_13).probabilities(
        object(), strict=True
    )

    assert distribution.local_values[0] is not None
    assert distribution.local_values[1] == 0.0
    assert distribution.local_values[2] is None


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


def test_subnormal_epsilon_and_rho_remain_representably_full_support() -> None:
    subnormal = math.nextafter(0.0, 1.0)
    distribution = _policy(
        calibration=CalibrationTuple(0.0, 0.0, subnormal, subnormal)
    ).probabilities(object(), strict=True)

    assert all(value > 0.0 for value in distribution.probabilities)
    assert math.fsum(distribution.probabilities) == pytest.approx(
        1.0, abs=3e-16
    )


def test_subnormal_positive_temperature_has_a_finite_limiting_softmax() -> None:
    subnormal = math.nextafter(0.0, 1.0)
    policy = _policy(
        outputs=torch.tensor([[-1.0, 3.0]]).repeat(5, 1),
        calibration=CalibrationTuple(0.0, subnormal, 0.0, 1.0),
    )

    distribution = policy.probabilities(object(), strict=True)

    assert distribution.probabilities[7] == 1.0
    assert all(math.isfinite(value) for value in distribution.probabilities)


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


def test_inverse_cdf_scales_by_fsum_instead_of_returning_a_zero_mass_tail() -> None:
    probabilities = [0.0] * 14
    probabilities[4] = 1.0 - 5e-13

    assert (
        stable_inverse_cdf(probabilities, math.nextafter(1.0, 0.0))
        is BidAction.BID_4
    )


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([0.0] * 13 + [1.0], BidAction.BID_13),
        ([0.0, 0.5, 0.0, 0.5] + [0.0] * 10, BidAction.BID_3),
        ([0.1] * 10 + [0.0] * 4, BidAction.BID_9),
    ],
)
def test_inverse_cdf_endpoint_returns_last_strictly_positive_action(
    probabilities: list[float], expected: BidAction
) -> None:
    assert stable_inverse_cdf(probabilities, math.nextafter(1.0, 0.0)) is expected


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
    ("outputs", "reason_fragment"),
    [
        (torch.zeros(4, 2), "shape"),
        (torch.full((5, 2), float("nan")), "finite"),
    ],
)
def test_formal_failures_raise_while_runtime_returns_structured_nsfp_fallback(
    outputs: torch.Tensor,
    reason_fragment: str,
) -> None:
    policy = _policy(outputs=outputs)
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


def test_policy_reconstructs_and_freezes_one_promoted_checkpoint_identity() -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    calibration = CalibrationTuple(0.5, 0.75, 0.2, 0.8)
    metadata = _promoted(calibration, ensemble)

    policy = _policy(ensemble=ensemble, metadata=metadata, calibration=calibration)

    assert policy.policy_id == metadata.policy_id
    assert policy.calibration == metadata.calibration
    assert policy.ensemble is ensemble
    assert ensemble.training is False
    assert all(parameter.requires_grad is False for parameter in ensemble.parameters())
    with pytest.raises(AttributeError):
        policy.calibration = CalibrationTuple(0.0, 0.0, 0.0, 1.0)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.policy_id = "f" * 64  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.metadata = replace(metadata, policy_id="f" * 64)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.ensemble = ResidualQEnsemble(SEEDS)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.nsfp = _FrozenNSFPDouble(_observation())  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda meta: replace(meta, policy_id="f" * 64), "metadata"),
        (
            lambda meta: replace(
                meta,
                calibration=CalibrationTuple(0.25, 0.75, 0.2, 0.8),
            ),
            "metadata",
        ),
        (lambda meta: replace(meta, member_init_seeds=(1, 2, 3, 4, 5)), "seed"),
        (lambda meta: replace(meta, play_pipeline_sha256="1" * 64), "metadata"),
    ],
)
def test_policy_rejects_policy_calibration_seed_and_provenance_drift(
    mutation: object, match: str
) -> None:
    calibration = CalibrationTuple(0.5, 0.75, 0.2, 0.8)
    metadata = _promoted(calibration)

    with pytest.raises(ValueError, match=match):
        _policy(
            calibration=calibration,
            metadata=mutation(metadata),  # type: ignore[operator]
        )


def test_policy_rejects_candidate_status() -> None:
    with pytest.raises(ValueError, match="promoted"):
        _policy(metadata=_SHARED_CANDIDATE)


def test_policy_rejects_ensemble_weight_drift() -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    calibration = CalibrationTuple(0.5, 0.75, 0.2, 0.8)
    metadata = _promoted(calibration, ensemble)
    with torch.no_grad():
        next(ensemble.parameters()).add_(1.0)

    with pytest.raises(ValueError, match="metadata"):
        _policy(ensemble=ensemble, metadata=metadata, calibration=calibration)


def test_policy_rejects_actual_loaded_nsfp_hash_drift() -> None:
    with pytest.raises(ValueError, match="NSFP"):
        _policy(checkpoint_nsfp_sha256="1" * 64)


def test_policy_requires_concrete_ensemble() -> None:
    metadata = _promoted(CalibrationTuple(0.5, 0.75, 0.2, 0.8))
    with pytest.raises(TypeError, match="ResidualQEnsemble"):
        StochasticResidualPolicy(
            _FrozenNSFPDouble(_observation()),
            nn.Identity(),  # type: ignore[arg-type]
            metadata,
        )


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


def test_fixed_policy_tape_empirically_matches_declared_distribution_via_public_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(calibration=CalibrationTuple(0.5, 0.75, 0.35, 0.9))
    frozen_distribution = BidDistribution(
        probabilities=EMPIRICAL_PROBABILITIES,
        center=BidAction.BID_6,
        local_values=(None, 0.0, None),
        policy_id=policy.policy_id,
    )
    monkeypatch.setattr(
        policy,
        "_distribution_from_observation",
        lambda observation: frozen_distribution,
    )
    legal = [to_local_bid(action) for action in BidAction]
    assert legal[0] == "nil"
    sample_size = 200_000
    counts = [0] * 14

    for index in range(sample_size):
        decision = policy.sample(object(), legal, _key(index), strict=True)
        counts[int(decision.action)] += 1

    frequencies = [count / sample_size for count in counts]
    assert all(
        abs(actual - expected) <= tolerance
        for actual, expected, tolerance in zip(
            frequencies,
            EMPIRICAL_PROBABILITIES,
            EMPIRICAL_TOLERANCES,
            strict=True,
        )
    )


def test_public_shapes_and_constructor_do_not_accept_rng_or_shuffle_seed() -> None:
    assert len(
        BidDistribution(
            tuple([1.0] + [0.0] * 13),
            BidAction.NIL,
            (None, 0.0, 0.0),
            "test-policy",
        ).probabilities
    ) == 14
    constructor = inspect.signature(StochasticResidualPolicy)
    assert list(constructor.parameters) == ["nsfp", "ensemble", "metadata"]
    assert "calibration" not in constructor.parameters
    assert "policy_id" not in constructor.parameters
    assert "rng" not in constructor.parameters
    assert "shuffle_seed" not in constructor.parameters
