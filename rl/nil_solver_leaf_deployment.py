"""Production loader for the converged four-role Nil solver-leaf actors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from rl.nil_solver_leaf_ppo import load_nil_role_actor_bundle
from rl.policy_network import PolicyMLP


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NIL_ACTOR_BUNDLE_PATH = (
    REPO_ROOT
    / "Spades_AI_GO-MCTS"
    / "checkpoints"
    / "nil_solver_leaf_four_role"
    / "actors_update_000020.json"
)
DEFAULT_NIL_ACTOR_BUNDLE_SHA256 = (
    "d0a61ce0a8253e109c4d3ac0b0eeb0447c01a952209fc38c7dd878264c8ce678"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DeployedNilActorBundle:
    """Hash-pinned, inference-checked production Nil policy bundle."""

    actors: dict[str, PolicyMLP]
    manifest: dict
    metadata: dict[str, dict]
    path: Path
    sha256: str
    model_id: str


def load_deployed_nil_actor_bundle(
    bundle_path: Path = DEFAULT_NIL_ACTOR_BUNDLE_PATH,
    *,
    expected_sha256: str = DEFAULT_NIL_ACTOR_BUNDLE_SHA256,
    device: torch.device | str = "cpu",
) -> DeployedNilActorBundle:
    """Load the selected production bundle and fail closed on any mismatch."""

    resolved = Path(bundle_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"deployed Nil actor bundle manifest not found: {resolved}"
        )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected Nil actor bundle SHA-256 must contain 64 hex digits")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "expected Nil actor bundle SHA-256 must contain 64 hex digits"
        ) from error

    actual_sha256 = _sha256(resolved)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "deployed Nil actor bundle SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    actors, manifest, metadata = load_nil_role_actor_bundle(
        resolved,
        device=device,
    )
    device_obj = torch.device(device)
    probe = torch.zeros(536, dtype=torch.float32, device=device_obj)
    with torch.inference_mode():
        for role, actor in actors.items():
            logits = actor(probe)
            if logits.shape != (52,) or not bool(torch.isfinite(logits).all().item()):
                raise ValueError(
                    f"deployed Nil actor {role!r} failed the inference smoke test"
                )

    update = int(manifest["training_update"])
    model_id = f"solver_leaf_nil_four_role_u{update}_{actual_sha256[:12]}"
    return DeployedNilActorBundle(
        actors=actors,
        manifest=manifest,
        metadata=metadata,
        path=resolved,
        sha256=actual_sha256,
        model_id=model_id,
    )


__all__ = [
    "DEFAULT_NIL_ACTOR_BUNDLE_PATH",
    "DEFAULT_NIL_ACTOR_BUNDLE_SHA256",
    "DeployedNilActorBundle",
    "load_deployed_nil_actor_bundle",
]
