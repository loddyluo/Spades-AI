"""Pinned production loader for the selected residual acting bidder."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.checkpoint import (
    BidderCheckpointMeta,
    CalibrationTuple,
    load_checkpoint,
    promote_meta,
)
from residual_bidder.config import BidderConfig, canonical_sha256
from residual_bidder.nsfp import FrozenNSFP
from residual_bidder.policy import BidDecision, StochasticResidualPolicy
from residual_bidder.random_tape import BidSamplingKey
from trick_taking.game_state import GameState


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "residual_bidder" / "base.yaml"
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_residual_100k.pt"
)
DEPLOYED_CHECKPOINT_SHA256 = (
    "633ad460961cf64c82a7c7966540d22d123e3461f72fb476ffad1aa4d91c0afc"
)
DEPLOYED_MODEL_ID = (
    "72b9b2fd95da2889e1cdd43527f9ed44d28b4490dc378e67d2924a3dbb9b5164"
)
DEPLOYED_CALIBRATION = CalibrationTuple(
    uncertainty_lambda=0.0,
    temperature=0.0,
    epsilon=0.0,
    rho=1.0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _play_pipeline_sha256(config: BidderConfig, root: Path) -> str:
    play_config = _resolve(root, config.play.config_path)
    actual_play_config_sha256 = _sha256(play_config)
    if actual_play_config_sha256 != config.play.config_sha256:
        raise ValueError(
            "deployed play config SHA-256 does not match residual bidder config"
        )
    return canonical_sha256(
        {
            "play_config_sha256": config.play.config_sha256,
            "source_manifest": [
                [source, _sha256(_resolve(root, source))]
                for source in config.play.source_manifest
            ],
        }
    )


def _checkpoint_dataset_sha256(checkpoint: Path) -> str:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
        raise ValueError("deployed checkpoint has no safe metadata mapping")
    value = artifact["metadata"].get("dataset_manifest_sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("deployed checkpoint has no valid dataset SHA-256")
    return value


def _promote_for_deployment(metadata: BidderCheckpointMeta) -> BidderCheckpointMeta:
    if metadata.status == "candidate":
        return promote_meta(metadata, DEPLOYED_CALIBRATION)
    if metadata.status == "promoted" and metadata.calibration == DEPLOYED_CALIBRATION:
        return metadata
    raise ValueError("deployed checkpoint has incompatible promotion metadata")


class DeployedActingBidder:
    """Read-only acting bidder; card-play belief remains the legacy NSFP model."""

    def __init__(
        self,
        policy: StochasticResidualPolicy,
        *,
        policy_seed: int,
        checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> None:
        if not isinstance(policy, StochasticResidualPolicy):
            raise TypeError("policy must be a StochasticResidualPolicy")
        if type(policy_seed) is not int:
            raise TypeError("policy_seed must be an integer")
        self._policy = policy
        self._policy_seed = policy_seed
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_sha256 = checkpoint_sha256

    @property
    def policy(self) -> StochasticResidualPolicy:
        return self._policy

    @property
    def model_id(self) -> str:
        return self._policy.metadata.model_id

    @property
    def policy_id(self) -> str:
        return self._policy.policy_id

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def policy_seed(self) -> int:
        return self._policy_seed

    def choose(
        self,
        state: GameState,
        legal_bids: Sequence[Any],
        *,
        logical_seat: int,
        deal_id: str,
        room_id: str,
    ) -> BidDecision:
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        if type(logical_seat) is not int or not 0 <= logical_seat < 4:
            raise ValueError("logical_seat must be in [0, 3]")
        if state.current_bidder != logical_seat:
            raise ValueError("acting bidder was called for the wrong seat")
        if not isinstance(deal_id, str) or not deal_id:
            raise ValueError("deal_id must be a nonempty string")
        if not isinstance(room_id, str) or not room_id:
            raise ValueError("room_id must be a nonempty string")
        bid_index = sum(not getattr(bid, "is_pass", False) for bid in state.bids)
        decision = self._policy.sample(
            state,
            legal_bids,
            BidSamplingKey(
                policy_seed=self._policy_seed,
                deal_id=deal_id,
                room_id=room_id,
                logical_seat=logical_seat,
                bid_index=bid_index,
            ),
            strict=False,
        )
        if not isinstance(decision.action, BidAction):
            raise TypeError("deployed policy returned a non-canonical action")
        local_bid = to_local_bid(decision.action)
        if local_bid not in legal_bids:
            raise ValueError(f"deployed policy returned illegal bid {local_bid!r}")
        return decision

    def describe(self) -> dict[str, object]:
        calibration = self._policy.calibration
        return {
            "name": "residual_q_100k",
            "model_id": self.model_id,
            "policy_id": self.policy_id,
            "checkpoint_sha256": self._checkpoint_sha256,
            "calibration": {
                "uncertainty_lambda": calibration.uncertainty_lambda,
                "temperature": calibration.temperature,
                "epsilon": calibration.epsilon,
                "rho": calibration.rho,
            },
            "belief_bidder": "bid_nsfp.pt",
        }


def load_deployed_acting_bidder(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    repo_root: Path = REPO_ROOT,
    device: str | torch.device = "cpu",
    policy_seed: int | None = None,
    expected_checkpoint_sha256: str = DEPLOYED_CHECKPOINT_SHA256,
    expected_model_id: str = DEPLOYED_MODEL_ID,
) -> DeployedActingBidder:
    """Load the selected 100k policy and validate every frozen dependency."""

    root = Path(repo_root).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config_file = Path(config_path).resolve()
    actual_checkpoint_sha256 = _sha256(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "deployed acting checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )

    config = BidderConfig.load(config_file)
    nsfp = FrozenNSFP.load(
        _resolve(root, config.nsfp.path),
        config.nsfp.sha256,
        torch.device(device),
    )
    ensemble, candidate_metadata = load_checkpoint(
        checkpoint,
        expected_nsfp_sha256=config.nsfp.sha256,
        expected_play_pipeline_sha256=_play_pipeline_sha256(config, root),
        expected_config_sha256=config.sha256(),
        expected_dataset_manifest_sha256=_checkpoint_dataset_sha256(checkpoint),
    )
    if candidate_metadata.model_id != expected_model_id:
        raise ValueError(
            "deployed acting checkpoint model_id mismatch: "
            f"expected {expected_model_id}, got {candidate_metadata.model_id}"
        )
    promoted = _promote_for_deployment(candidate_metadata)
    policy = StochasticResidualPolicy(nsfp, ensemble, promoted)
    resolved_seed = config.policy.policy_seed if policy_seed is None else policy_seed
    return DeployedActingBidder(
        policy,
        policy_seed=resolved_seed,
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_checkpoint_sha256,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEPLOYED_CALIBRATION",
    "DEPLOYED_CHECKPOINT_SHA256",
    "DEPLOYED_MODEL_ID",
    "DeployedActingBidder",
    "load_deployed_acting_bidder",
]
