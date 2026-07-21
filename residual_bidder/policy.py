"""Calibrated residual-bid distributions and deterministic sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from residual_bidder.actions import BidAction, neighborhood, to_local_bid
from residual_bidder.checkpoint import CalibrationTuple
from residual_bidder.model import ENSEMBLE_MEMBERS, build_residual_input
from residual_bidder.nsfp import FrozenNSFP, NSFPObservation
from residual_bidder.random_tape import BidSamplingKey, policy_uniform
from trick_taking.game_state import GameState


ACTION_COUNT = len(BidAction)
LEGACY_POLICY_ID = "legacy-nsfp"
FALLBACK_POLICY_ID = "legacy-nsfp-fallback"


@dataclass(frozen=True)
class BidDistribution:
    """A complete canonical bidding distribution and its local diagnostics."""

    probabilities: tuple[float, ...]
    center: BidAction
    local_values: tuple[float | None, float, float | None]
    policy_id: str

    def __post_init__(self) -> None:
        if len(self.probabilities) != ACTION_COUNT:
            raise ValueError(f"probabilities must contain exactly {ACTION_COUNT} values")
        if not isinstance(self.center, BidAction):
            raise TypeError("center must be a BidAction")
        if len(self.local_values) != 3 or self.local_values[1] != 0.0:
            raise ValueError("local_values must be (lower, exactly-zero center, upper)")
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a nonempty string")
        if any(not math.isfinite(value) or value < 0.0 for value in self.probabilities):
            raise ValueError("probabilities must be finite and nonnegative")
        if not math.isclose(math.fsum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        if any(
            value is not None and not math.isfinite(value) for value in self.local_values
        ):
            raise ValueError("local_values must be finite where present")


@dataclass(frozen=True)
class BidDecision:
    """A sampled canonical action with complete runtime provenance."""

    action: BidAction
    distribution: BidDistribution
    uniform: float
    effective_policy_id: str
    fallback_reason: str | None


@runtime_checkable
class ActingBidPolicy(Protocol):
    """Common acting interface for legacy and stochastic bidders."""

    policy_id: str

    def probabilities(self, state: GameState, *, strict: bool) -> BidDistribution: ...

    def sample(
        self,
        state: GameState,
        legal_bids: Sequence[object],
        key: BidSamplingKey,
        *,
        strict: bool,
    ) -> BidDecision: ...


def geometric_tail(center: BidAction, rho: float) -> torch.Tensor:
    """Return the normalized geometric tail over all canonical actions."""

    if not isinstance(center, BidAction):
        raise TypeError("center must be a BidAction")
    if type(rho) not in (int, float) or not math.isfinite(float(rho)) or not 0 < rho <= 1:
        raise ValueError("rho must be finite and in (0, 1]")
    indices = torch.arange(ACTION_COUNT, dtype=torch.float64)
    weights = torch.pow(float(rho), torch.abs(indices - int(center)))
    return weights / weights.sum(dtype=torch.float64)


def stable_inverse_cdf(probabilities: Sequence[float], u: float) -> BidAction:
    """Sample in canonical action order using a supplied open-interval uniform."""

    if not isinstance(probabilities, Sequence) or isinstance(
        probabilities, (str, bytes)
    ):
        raise TypeError("probabilities must be a sequence")
    if len(probabilities) != ACTION_COUNT:
        raise ValueError(f"probabilities must contain exactly {ACTION_COUNT} values")
    values = tuple(float(value) for value in probabilities)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")
    if type(u) not in (int, float) or not math.isfinite(float(u)) or not 0.0 < u < 1.0:
        raise ValueError("u must be finite and strictly between zero and one")

    cumulative = 0.0
    for action, probability in zip(BidAction, values, strict=True):
        cumulative += probability
        if u <= cumulative:
            return action
    return BidAction.BID_13


def _one_hot_distribution(
    center: BidAction, policy_id: str
) -> BidDistribution:
    probabilities = tuple(1.0 if action is center else 0.0 for action in BidAction)
    return BidDistribution(
        probabilities=probabilities,
        center=center,
        local_values=(None, 0.0, None),
        policy_id=policy_id,
    )


def _contains_expected_bid(legal_bids: Sequence[object], action: BidAction) -> bool:
    if not isinstance(legal_bids, Sequence) or isinstance(legal_bids, (str, bytes)):
        raise TypeError("legal_bids must be a sequence")
    expected = to_local_bid(action)
    return any(isinstance(candidate, str) and candidate == expected for candidate in legal_bids)


class NSFPArgmaxPolicy:
    """The deterministic normalized NSFP argmax acting policy."""

    def __init__(self, nsfp: FrozenNSFP, policy_id: str = LEGACY_POLICY_ID) -> None:
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a nonempty string")
        self.nsfp = nsfp
        self.policy_id = policy_id

    def probabilities(self, state: GameState, *, strict: bool) -> BidDistribution:
        del strict
        observation = self.nsfp.observe(state)
        return _one_hot_distribution(observation.center, self.policy_id)

    def probabilities_batch(
        self, states: Sequence[GameState], *, strict: bool
    ) -> list[BidDistribution]:
        return [self.probabilities(state, strict=strict) for state in states]

    def sample(
        self,
        state: GameState,
        legal_bids: Sequence[object],
        key: BidSamplingKey,
        *,
        strict: bool,
    ) -> BidDecision:
        distribution = self.probabilities(state, strict=strict)
        uniform = policy_uniform(key)
        if not _contains_expected_bid(legal_bids, distribution.center):
            raise ValueError(
                f"expected legal bid {to_local_bid(distribution.center)!r} is absent"
            )
        return BidDecision(
            action=distribution.center,
            distribution=distribution,
            uniform=uniform,
            effective_policy_id=self.policy_id,
            fallback_reason=None,
        )


class StochasticResidualPolicy:
    """A calibrated residual-Q policy with fail-closed formal semantics."""

    def __init__(
        self,
        nsfp: FrozenNSFP,
        ensemble: torch.nn.Module,
        calibration: CalibrationTuple,
        policy_id: str,
        *,
        expected_nsfp_sha256: str,
        checkpoint_nsfp_sha256: str,
    ) -> None:
        self.nsfp = nsfp
        self.ensemble = ensemble
        self.calibration = calibration
        self.policy_id = policy_id
        self.expected_nsfp_sha256 = expected_nsfp_sha256
        self.checkpoint_nsfp_sha256 = checkpoint_nsfp_sha256

    def _validate_configuration(self) -> None:
        calibration = self.calibration
        if not isinstance(calibration, CalibrationTuple):
            raise ValueError("calibration must be a CalibrationTuple")
        values = (
            calibration.uncertainty_lambda,
            calibration.temperature,
            calibration.epsilon,
            calibration.rho,
        )
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("calibration fields must be finite numbers")
        if calibration.uncertainty_lambda < 0 or calibration.temperature < 0:
            raise ValueError("calibration lambda and temperature must be nonnegative")
        if not 0 <= calibration.epsilon <= 1 or not 0 < calibration.rho <= 1:
            raise ValueError("calibration epsilon or rho is outside its allowed range")
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a nonempty string")
        for name, digest in (
            ("expected NSFP hash", self.expected_nsfp_sha256),
            ("checkpoint NSFP hash", self.checkpoint_nsfp_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.checkpoint_nsfp_sha256 != self.expected_nsfp_sha256:
            raise ValueError("checkpoint NSFP hash does not match the frozen expected hash")

    def _distribution_from_observation(
        self, observation: NSFPObservation
    ) -> BidDistribution:
        """Construct the distribution used by every public inference operation."""

        self._validate_configuration()
        residual_input = build_residual_input(observation)
        with torch.inference_mode():
            outputs = self.ensemble(residual_input.values)
        if not isinstance(outputs, torch.Tensor) or outputs.shape != (ENSEMBLE_MEMBERS, 2):
            shape = None if not isinstance(outputs, torch.Tensor) else tuple(outputs.shape)
            raise ValueError(f"residual ensemble output must have shape (5, 2), got {shape}")
        if not torch.is_floating_point(outputs):
            raise ValueError("residual ensemble output must be floating point")
        if not bool(torch.isfinite(outputs).all().item()):
            raise ValueError("residual ensemble output must be finite")

        outputs64 = outputs.to(dtype=torch.float64)
        means = outputs64.mean(dim=0)
        stds = outputs64.std(dim=0, unbiased=False)
        adjusted = means - float(self.calibration.uncertainty_lambda) * stds
        local = neighborhood(observation.center)
        lower_value = None if local.lower is None else adjusted[0]
        upper_value = None if local.upper is None else adjusted[1]
        center_value = torch.zeros((), dtype=torch.float64, device=outputs.device)
        stable_candidates: list[tuple[BidAction, torch.Tensor]] = [
            (local.center, center_value)
        ]
        if lower_value is not None:
            stable_candidates.append((local.lower, lower_value))
        if upper_value is not None:
            stable_candidates.append((local.upper, upper_value))

        local_core = torch.zeros(ACTION_COUNT, dtype=torch.float64, device=outputs.device)
        if self.calibration.temperature == 0:
            stable_values = torch.stack([value for _, value in stable_candidates])
            selected_index = int(torch.argmax(stable_values).item())
            selected_action = stable_candidates[selected_index][0]
            local_core[int(selected_action)] = 1.0
        else:
            ordered_candidates: list[tuple[BidAction, torch.Tensor]] = []
            if lower_value is not None:
                ordered_candidates.append((local.lower, lower_value))
            ordered_candidates.append((local.center, center_value))
            if upper_value is not None:
                ordered_candidates.append((local.upper, upper_value))
            candidate_values = torch.stack([value for _, value in ordered_candidates])
            masses = torch.softmax(
                candidate_values / float(self.calibration.temperature), dim=0
            )
            for (action, _), mass in zip(ordered_candidates, masses, strict=True):
                local_core[int(action)] = mass

        tail = geometric_tail(observation.center, self.calibration.rho).to(outputs.device)
        final = (
            (1.0 - float(self.calibration.epsilon)) * local_core
            + float(self.calibration.epsilon) * tail
        )
        final /= final.sum(dtype=torch.float64)
        if not bool(torch.isfinite(final).all().item()) or bool((final < 0).any().item()):
            raise ValueError("final policy probabilities must be finite and nonnegative")

        return BidDistribution(
            probabilities=tuple(float(value) for value in final.cpu().tolist()),
            center=observation.center,
            local_values=(
                None if lower_value is None else float(lower_value.item()),
                0.0,
                None if upper_value is None else float(upper_value.item()),
            ),
            policy_id=self.policy_id,
        )

    @staticmethod
    def _fallback_distribution(observation: NSFPObservation) -> BidDistribution:
        return _one_hot_distribution(observation.center, FALLBACK_POLICY_ID)

    def probabilities(self, state: GameState, *, strict: bool) -> BidDistribution:
        observation = self.nsfp.observe(state)
        try:
            return self._distribution_from_observation(observation)
        except (RuntimeError, TypeError, ValueError):
            if strict:
                raise
            return self._fallback_distribution(observation)

    def probabilities_batch(
        self, states: Sequence[GameState], *, strict: bool
    ) -> list[BidDistribution]:
        if not isinstance(states, Sequence):
            raise TypeError("states must be a sequence")
        return [self.probabilities(state, strict=strict) for state in states]

    def _fallback_decision(
        self,
        observation: NSFPObservation,
        legal_bids: Sequence[object],
        uniform: float,
        reason: str,
    ) -> BidDecision:
        if not _contains_expected_bid(legal_bids, observation.center):
            raise ValueError(
                f"expected fallback legal bid {to_local_bid(observation.center)!r} is absent"
            )
        distribution = self._fallback_distribution(observation)
        return BidDecision(
            action=observation.center,
            distribution=distribution,
            uniform=uniform,
            effective_policy_id=FALLBACK_POLICY_ID,
            fallback_reason=reason,
        )

    def sample(
        self,
        state: GameState,
        legal_bids: Sequence[object],
        key: BidSamplingKey,
        *,
        strict: bool,
    ) -> BidDecision:
        uniform = policy_uniform(key)
        observation = self.nsfp.observe(state)
        try:
            distribution = self._distribution_from_observation(observation)
        except (RuntimeError, TypeError, ValueError) as error:
            if strict:
                raise
            reason = f"residual-policy-error:{type(error).__name__}:{error}"
            return self._fallback_decision(observation, legal_bids, uniform, reason)

        action = stable_inverse_cdf(distribution.probabilities, uniform)
        if not _contains_expected_bid(legal_bids, action):
            error = ValueError(f"expected legal bid {to_local_bid(action)!r} is absent")
            if strict:
                raise error
            reason = f"legal-action-drift:{type(error).__name__}:{error}"
            return self._fallback_decision(observation, legal_bids, uniform, reason)
        return BidDecision(
            action=action,
            distribution=distribution,
            uniform=uniform,
            effective_policy_id=self.policy_id,
            fallback_reason=None,
        )
