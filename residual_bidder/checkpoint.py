"""Content-addressed, fail-closed checkpoints for residual Q ensembles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from residual_bidder.model import ENSEMBLE_MEMBERS, ResidualQEnsemble


CHECKPOINT_SCHEMA = "residual-q-ensemble-v1"
MODEL_ID_HASH_FORMAT = "sha256-framed-little-endian-v1"
_META_FIELDS = {
    "schema",
    "status",
    "model_id",
    "policy_id",
    "iteration",
    "nsfp_sha256",
    "play_pipeline_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
    "member_init_seeds",
    "calibration",
}
_HASH_FIELDS = (
    "nsfp_sha256",
    "play_pipeline_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
)


@dataclass(frozen=True)
class CalibrationTuple:
    uncertainty_lambda: float
    temperature: float
    epsilon: float
    rho: float


@dataclass(frozen=True)
class BidderCheckpointMeta:
    schema: str
    status: Literal["candidate", "promoted"]
    model_id: str
    policy_id: str | None
    iteration: int
    nsfp_sha256: str
    play_pipeline_sha256: str
    config_sha256: str
    dataset_manifest_sha256: str
    member_init_seeds: tuple[int, int, int, int, int]
    calibration: CalibrationTuple | None


class CheckpointDurabilityError(OSError):
    """Publication succeeded, but parent-directory durability was not confirmed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _calibration_dict(calibration: CalibrationTuple) -> dict[str, float]:
    return {
        "uncertainty_lambda": float(calibration.uncertainty_lambda),
        "temperature": float(calibration.temperature),
        "epsilon": float(calibration.epsilon),
        "rho": float(calibration.rho),
    }


def _metadata_dict(metadata: BidderCheckpointMeta) -> dict[str, object]:
    return {
        "schema": metadata.schema,
        "status": metadata.status,
        "model_id": metadata.model_id,
        "policy_id": metadata.policy_id,
        "iteration": metadata.iteration,
        "nsfp_sha256": metadata.nsfp_sha256,
        "play_pipeline_sha256": metadata.play_pipeline_sha256,
        "config_sha256": metadata.config_sha256,
        "dataset_manifest_sha256": metadata.dataset_manifest_sha256,
        "member_init_seeds": list(metadata.member_init_seeds),
        "calibration": (
            None if metadata.calibration is None else _calibration_dict(metadata.calibration)
        ),
    }


def _parse_calibration(raw: object) -> CalibrationTuple | None:
    if raw is None:
        return None
    fields = {"uncertainty_lambda", "temperature", "epsilon", "rho"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("calibration must have exactly the four calibrated fields")
    if any(type(raw[field]) not in (int, float) for field in fields):
        raise ValueError("calibration fields must be numbers")
    calibration = CalibrationTuple(**{field: float(raw[field]) for field in fields})
    _validate_calibration(calibration)
    return calibration


def _parse_metadata(raw: object) -> BidderCheckpointMeta:
    if not isinstance(raw, dict) or set(raw) != _META_FIELDS:
        raise ValueError("checkpoint metadata fields do not match the frozen schema")
    seeds = raw["member_init_seeds"]
    if not isinstance(seeds, list) or len(seeds) != ENSEMBLE_MEMBERS:
        raise ValueError("checkpoint must name exactly five member seeds")
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("member seeds must be integers")
    metadata = BidderCheckpointMeta(
        schema=raw["schema"],
        status=raw["status"],
        model_id=raw["model_id"],
        policy_id=raw["policy_id"],
        iteration=raw["iteration"],
        nsfp_sha256=raw["nsfp_sha256"],
        play_pipeline_sha256=raw["play_pipeline_sha256"],
        config_sha256=raw["config_sha256"],
        dataset_manifest_sha256=raw["dataset_manifest_sha256"],
        member_init_seeds=tuple(seeds),
        calibration=_parse_calibration(raw["calibration"]),
    )
    _validate_metadata_shape(metadata)
    return metadata


def _validate_calibration(calibration: CalibrationTuple) -> None:
    values = _calibration_dict(calibration)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
        raise ValueError("calibration fields must be finite numbers")
    if calibration.uncertainty_lambda < 0 or calibration.temperature < 0:
        raise ValueError("lambda and temperature must be nonnegative")
    if not 0 <= calibration.epsilon <= 1 or not 0 < calibration.rho <= 1:
        raise ValueError("epsilon and rho are outside their allowed ranges")


def _validate_metadata_shape(metadata: BidderCheckpointMeta) -> None:
    if metadata.schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unknown checkpoint schema: {metadata.schema!r}")
    if metadata.status not in ("candidate", "promoted"):
        raise ValueError(f"unknown checkpoint status: {metadata.status!r}")
    if type(metadata.iteration) is not int or metadata.iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    for field in _HASH_FIELDS:
        if not _is_sha256(getattr(metadata, field)):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if not _is_sha256(metadata.model_id):
        raise ValueError("model_id must be a lowercase SHA-256 digest")
    if (
        not isinstance(metadata.member_init_seeds, tuple)
        or len(metadata.member_init_seeds) != ENSEMBLE_MEMBERS
        or any(type(seed) is not int for seed in metadata.member_init_seeds)
        or len(set(metadata.member_init_seeds)) != ENSEMBLE_MEMBERS
    ):
        raise ValueError("member_init_seeds must contain five distinct integers")
    if metadata.status == "candidate":
        if metadata.policy_id is not None or metadata.calibration is not None:
            raise ValueError("candidate checkpoints cannot have a policy ID or calibration")
    else:
        if not _is_sha256(metadata.policy_id) or metadata.calibration is None:
            raise ValueError("promoted checkpoints require calibration and a policy ID")
        _validate_calibration(metadata.calibration)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _hash_frame(digest: object, domain: bytes, payload: bytes) -> None:
    """Add an explicitly separated and length-framed field to a content digest."""

    if not isinstance(domain, bytes) or not isinstance(payload, bytes):
        raise TypeError("hash frames require byte domains and payloads")
    digest.update(struct.pack("<I", len(domain)))
    digest.update(domain)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _tensor_bytes_little_endian(tensor: torch.Tensor) -> bytes:
    """Return canonical contiguous tensor bytes in documented little-endian order."""

    array = tensor.detach().cpu().contiguous().numpy()
    little_endian_dtype = array.dtype.newbyteorder("<")
    return array.astype(little_endian_dtype, copy=False).tobytes(order="C")


def _model_id(
    state_dict: Mapping[str, torch.Tensor], metadata: BidderCheckpointMeta
) -> str:
    digest = hashlib.sha256()
    identity = {
        "schema": metadata.schema,
        "iteration": metadata.iteration,
        "nsfp_sha256": metadata.nsfp_sha256,
        "play_pipeline_sha256": metadata.play_pipeline_sha256,
        "config_sha256": metadata.config_sha256,
        "dataset_manifest_sha256": metadata.dataset_manifest_sha256,
        "member_init_seeds": list(metadata.member_init_seeds),
    }
    _hash_frame(digest, b"hash-format", MODEL_ID_HASH_FORMAT.encode("ascii"))
    _hash_frame(digest, b"model-metadata", _canonical_json(identity))
    for name in sorted(state_dict):
        tensor = state_dict[name]
        shape = struct.pack("<Q", tensor.ndim) + b"".join(
            struct.pack("<Q", dimension) for dimension in tensor.shape
        )
        _hash_frame(digest, b"tensor-name", name.encode("utf-8"))
        _hash_frame(digest, b"tensor-dtype", f"{tensor.dtype}-little-endian".encode("ascii"))
        _hash_frame(digest, b"tensor-shape", shape)
        _hash_frame(digest, b"tensor-bytes", _tensor_bytes_little_endian(tensor))
    return digest.hexdigest()


def _policy_id(metadata: BidderCheckpointMeta, calibration: CalibrationTuple) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "model_id": metadata.model_id,
                "nsfp_sha256": metadata.nsfp_sha256,
                "calibration": _calibration_dict(calibration),
            }
        )
    ).hexdigest()


def _state_dict_copy(ensemble: ResidualQEnsemble) -> dict[str, torch.Tensor]:
    if not isinstance(ensemble, ResidualQEnsemble) or len(ensemble.members) != ENSEMBLE_MEMBERS:
        raise ValueError("checkpoint model must be a five-member ResidualQEnsemble")
    state_dict = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in ensemble.state_dict().items()
    }
    if any(
        not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all().item())
        for tensor in state_dict.values()
    ):
        raise ValueError("checkpoint parameters must be finite floating-point tensors")
    return state_dict


def build_candidate_meta(
    ensemble: ResidualQEnsemble,
    *,
    iteration: int,
    nsfp_sha256: str,
    play_pipeline_sha256: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    member_init_seeds: tuple[int, int, int, int, int],
) -> BidderCheckpointMeta:
    """Create an uncalibrated candidate identity from weights and frozen provenance."""

    state_dict = _state_dict_copy(ensemble)
    provisional = BidderCheckpointMeta(
        schema=CHECKPOINT_SCHEMA,
        status="candidate",
        model_id="0" * 64,
        policy_id=None,
        iteration=iteration,
        nsfp_sha256=nsfp_sha256,
        play_pipeline_sha256=play_pipeline_sha256,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        member_init_seeds=member_init_seeds,
        calibration=None,
    )
    _validate_metadata_shape(provisional)
    if ensemble.member_init_seeds != member_init_seeds:
        raise ValueError("metadata member seeds do not match ensemble construction seeds")
    return BidderCheckpointMeta(
        schema=provisional.schema,
        status=provisional.status,
        model_id=_model_id(state_dict, provisional),
        policy_id=None,
        iteration=provisional.iteration,
        nsfp_sha256=provisional.nsfp_sha256,
        play_pipeline_sha256=provisional.play_pipeline_sha256,
        config_sha256=provisional.config_sha256,
        dataset_manifest_sha256=provisional.dataset_manifest_sha256,
        member_init_seeds=provisional.member_init_seeds,
        calibration=None,
    )


def promote_meta(
    candidate: BidderCheckpointMeta, calibration: CalibrationTuple
) -> BidderCheckpointMeta:
    """Derive the immutable policy identity for a calibrated candidate."""

    _validate_metadata_shape(candidate)
    if candidate.status != "candidate":
        raise ValueError("only a candidate checkpoint can be promoted")
    if not isinstance(calibration, CalibrationTuple):
        raise TypeError("calibration must be a CalibrationTuple")
    _validate_calibration(calibration)
    return BidderCheckpointMeta(
        schema=candidate.schema,
        status="promoted",
        model_id=candidate.model_id,
        policy_id=_policy_id(candidate, calibration),
        iteration=candidate.iteration,
        nsfp_sha256=candidate.nsfp_sha256,
        play_pipeline_sha256=candidate.play_pipeline_sha256,
        config_sha256=candidate.config_sha256,
        dataset_manifest_sha256=candidate.dataset_manifest_sha256,
        member_init_seeds=candidate.member_init_seeds,
        calibration=calibration,
    )


def _validate_identity(
    state_dict: Mapping[str, torch.Tensor], metadata: BidderCheckpointMeta
) -> None:
    if metadata.model_id != _model_id(state_dict, metadata):
        raise ValueError("checkpoint model_id does not match its content")
    if metadata.status == "promoted":
        assert metadata.calibration is not None
        if metadata.policy_id != _policy_id(metadata, metadata.calibration):
            raise ValueError("checkpoint policy_id does not match its calibrated content")


def _validate_raw_state_dict(
    raw_state_dict: object, expected_state_dict: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Validate serialized tensors exactly before any module load can normalize them."""

    if not isinstance(raw_state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in raw_state_dict.items()
    ):
        raise ValueError("checkpoint state_dict must map strings only to tensors")
    if set(raw_state_dict) != set(expected_state_dict):
        raise ValueError("checkpoint state_dict member keys do not match the exact architecture")

    validated: dict[str, torch.Tensor] = {}
    for name in sorted(expected_state_dict):
        tensor = raw_state_dict[name]
        expected = expected_state_dict[name]
        if tensor.shape != expected.shape:
            raise ValueError(f"checkpoint tensor {name!r} has the wrong shape")
        if tensor.dtype != expected.dtype:
            raise ValueError(f"checkpoint tensor {name!r} has the wrong dtype")
        if tensor.layout != torch.strided or tensor.layout != expected.layout:
            raise ValueError(f"checkpoint tensor {name!r} has the wrong layout")
        if tensor.device != expected.device or tensor.device.type != "cpu":
            raise ValueError(f"checkpoint tensor {name!r} must be on CPU")
        if tensor.stride() != expected.stride():
            raise ValueError(f"checkpoint tensor {name!r} has the wrong stride")
        if tensor.storage_offset() != expected.storage_offset():
            raise ValueError(f"checkpoint tensor {name!r} has the wrong storage offset")
        if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"checkpoint tensor {name!r} must be finite floating point")
        validated[name] = tensor
    return validated


def _load_and_validate_artifact(
    path: Path,
    *,
    expected_nsfp_sha256: str,
    expected_play_pipeline_sha256: str | None,
    expected_config_sha256: str,
    expected_dataset_manifest_sha256: str,
    require_promoted: bool,
) -> tuple[ResidualQEnsemble, BidderCheckpointMeta]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"checkpoint could not be loaded safely: {error}") from error
    if not isinstance(artifact, dict) or set(artifact) != {"metadata", "state_dict"}:
        raise ValueError("checkpoint artifact fields do not match the frozen schema")
    metadata = _parse_metadata(artifact["metadata"])
    expected = {
        "nsfp_sha256": expected_nsfp_sha256,
        "play_pipeline_sha256": expected_play_pipeline_sha256,
        "config_sha256": expected_config_sha256,
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
    }
    for field, value in expected.items():
        # The card-play pipeline is training provenance, not part of the
        # acting bidder's runtime interface. Production callers may therefore
        # leave this expectation unset, while reproducibility-sensitive
        # training/evaluation callers can still require an exact match.
        if value is None:
            continue
        if not _is_sha256(value):
            raise ValueError(f"expected_{field} must be a lowercase SHA-256 digest")
        if getattr(metadata, field) != value:
            raise ValueError(f"checkpoint {field} does not match the frozen expected hash")
    if require_promoted and metadata.status != "promoted":
        raise ValueError("a promoted checkpoint is required")

    ensemble = ResidualQEnsemble(metadata.member_init_seeds)
    raw_state_dict = _validate_raw_state_dict(
        artifact["state_dict"], ensemble.state_dict()
    )
    _validate_identity(raw_state_dict, metadata)
    try:
        ensemble.load_state_dict(raw_state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint model dimensions or member count are invalid: {error}") from error
    return ensemble, metadata


def _fsync_parent_directory(parent: Path) -> None:
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_checkpoint(
    path: Path,
    *,
    expected_nsfp_sha256: str,
    expected_play_pipeline_sha256: str | None,
    expected_config_sha256: str,
    expected_dataset_manifest_sha256: str,
    require_promoted: bool = False,
) -> tuple[ResidualQEnsemble, BidderCheckpointMeta]:
    """Safely load a checkpoint, optionally enforcing card-play provenance."""

    return _load_and_validate_artifact(
        Path(path),
        expected_nsfp_sha256=expected_nsfp_sha256,
        expected_play_pipeline_sha256=expected_play_pipeline_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        require_promoted=require_promoted,
    )


def save_checkpoint_atomic(
    path: Path, ensemble: ResidualQEnsemble, metadata: BidderCheckpointMeta
) -> None:
    """Safely validate a sibling temp, atomically replace, and fsync the directory.

    If the final directory fsync fails, replacement has already happened. The visible
    destination is the fully validated new artifact, while crash persistence is not
    confirmed and may resolve to either the old or new directory entry.
    """

    path = Path(path)
    _validate_metadata_shape(metadata)
    if ensemble.member_init_seeds != metadata.member_init_seeds:
        raise ValueError("metadata member seeds do not match ensemble construction seeds")
    state_dict = _state_dict_copy(ensemble)
    _validate_identity(state_dict, metadata)
    artifact = {
        "metadata": _metadata_dict(metadata),
        "state_dict": state_dict,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(artifact, handle)
            handle.flush()
            os.fsync(handle.fileno())

        validated_ensemble, validated_metadata = _load_and_validate_artifact(
            temporary_path,
            expected_nsfp_sha256=metadata.nsfp_sha256,
            expected_play_pipeline_sha256=metadata.play_pipeline_sha256,
            expected_config_sha256=metadata.config_sha256,
            expected_dataset_manifest_sha256=metadata.dataset_manifest_sha256,
            require_promoted=metadata.status == "promoted",
        )
        if validated_metadata != metadata:
            raise ValueError("temporary checkpoint metadata changed during validation")
        for name, tensor in artifact["state_dict"].items():
            if not torch.equal(validated_ensemble.state_dict()[name], tensor):
                raise ValueError("temporary checkpoint tensor changed during validation")

        os.replace(temporary_path, path)
        temporary_path = None
        try:
            _fsync_parent_directory(path.parent)
        except OSError as error:
            raise CheckpointDurabilityError(
                "checkpoint was atomically replaced, but directory durability is unconfirmed"
            ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# Concise aliases for callers that use generic checkpoint terminology.
save_checkpoint = save_checkpoint_atomic
