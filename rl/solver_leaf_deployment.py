"""Hash-pinned production loader for the non-Nil solver-leaf actor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from rl.first4_observation import ENCODER_SCHEMA, FirstFourFeatureEncoderV2
from rl.policy_network import PolicyMLP
from rl.solver_leaf_ppo import actor_metadata_path, load_exported_actor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVER_LEAF_ACTOR_PATH = (
    REPO_ROOT
    / "Spades_AI_GO-MCTS"
    / "checkpoints"
    / "solver_leaf_nonnil"
    / "actor_update_000020.pt"
)
DEFAULT_SOLVER_LEAF_ACTOR_SHA256 = (
    "e5028b0264271633a3b10636970aea0c91fabf33771ddb12c79f0df898c5a92e"
)
DEFAULT_SOLVER_LEAF_SIDECAR_SHA256 = (
    "02a5a7309441228efb207d98f2fc42c3883ad6cfd36572d9c61f93deba775d20"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"expected {label} SHA-256 must contain 64 hex digits")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"expected {label} SHA-256 must contain 64 hex digits"
        ) from error
    return value.lower()


@dataclass(frozen=True, slots=True)
class DeployedSolverLeafActor:
    """Validated production policy and its immutable identity."""

    actor: PolicyMLP
    metadata: dict
    path: Path
    sha256: str
    sidecar_sha256: str
    model_id: str


def load_deployed_solver_leaf_actor(
    actor_path: Path = DEFAULT_SOLVER_LEAF_ACTOR_PATH,
    *,
    expected_sha256: str = DEFAULT_SOLVER_LEAF_ACTOR_SHA256,
    expected_sidecar_sha256: str = DEFAULT_SOLVER_LEAF_SIDECAR_SHA256,
    device: torch.device | str = "cpu",
) -> DeployedSolverLeafActor:
    """Load the selected non-Nil actor and fail closed on any mismatch."""

    resolved = Path(actor_path).expanduser().resolve()
    sidecar = actor_metadata_path(resolved)
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            "deployed solver-leaf actor weights and metadata sidecar are both "
            f"required: {resolved}"
        )

    expected_actor = _validated_sha256(expected_sha256, label="actor")
    expected_sidecar = _validated_sha256(
        expected_sidecar_sha256,
        label="actor sidecar",
    )
    actual_actor = _sha256(resolved)
    actual_sidecar = _sha256(sidecar)
    if actual_actor != expected_actor:
        raise ValueError(
            "deployed solver-leaf actor SHA-256 mismatch: "
            f"expected {expected_actor}, got {actual_actor}"
        )
    if actual_sidecar != expected_sidecar:
        raise ValueError(
            "deployed solver-leaf actor sidecar SHA-256 mismatch: "
            f"expected {expected_sidecar}, got {actual_sidecar}"
        )

    actor, metadata = load_exported_actor(resolved, device=device)
    if metadata.get("schema") != "solver-leaf-actor-v1":
        raise ValueError("deployed solver-leaf actor uses an unsupported schema")
    if metadata.get("encoder_schema") != ENCODER_SCHEMA:
        raise ValueError("deployed solver-leaf actor uses an incompatible encoder")
    if metadata.get("input_dim") != FirstFourFeatureEncoderV2.TOTAL_DIM:
        raise ValueError("deployed solver-leaf actor has an incompatible input size")
    if metadata.get("output_dim") != 52:
        raise ValueError("deployed solver-leaf actor has an incompatible output size")

    device_obj = torch.device(device)
    probe = torch.zeros(
        FirstFourFeatureEncoderV2.TOTAL_DIM,
        dtype=torch.float32,
        device=device_obj,
    )
    with torch.inference_mode():
        logits = actor(probe)
    if logits.shape != (52,) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("deployed solver-leaf actor failed the inference smoke test")

    update = int(metadata["training_update"])
    model_id = f"solver_leaf_nonnil_u{update}_{actual_actor[:12]}"
    return DeployedSolverLeafActor(
        actor=actor,
        metadata=metadata,
        path=resolved,
        sha256=actual_actor,
        sidecar_sha256=actual_sidecar,
        model_id=model_id,
    )


__all__ = [
    "DEFAULT_SOLVER_LEAF_ACTOR_PATH",
    "DEFAULT_SOLVER_LEAF_ACTOR_SHA256",
    "DEFAULT_SOLVER_LEAF_SIDECAR_SHA256",
    "DeployedSolverLeafActor",
    "load_deployed_solver_leaf_actor",
]
