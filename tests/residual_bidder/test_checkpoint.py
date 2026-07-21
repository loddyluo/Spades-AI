from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import residual_bidder.checkpoint as checkpoint_module
from residual_bidder.checkpoint import (
    CHECKPOINT_SCHEMA,
    BidderCheckpointMeta,
    CalibrationTuple,
    build_candidate_meta,
    load_checkpoint,
    promote_meta,
    save_checkpoint_atomic,
)
from residual_bidder.model import ResidualQEnsemble


HASHES = {
    "nsfp_sha256": "1" * 64,
    "play_pipeline_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "dataset_manifest_sha256": "4" * 64,
}
SEEDS = (1701, 1702, 1703, 1704, 1705)


def _candidate(ensemble: ResidualQEnsemble) -> BidderCheckpointMeta:
    return build_candidate_meta(ensemble, iteration=3, member_init_seeds=SEEDS, **HASHES)


def _expected_hashes() -> dict[str, str]:
    return {f"expected_{name}": value for name, value in HASHES.items()}


def _transaction_paths(destination: Path) -> dict[str, Path]:
    return {
        "backup": destination.with_name(f".{destination.name}.rollback"),
        "absent": destination.with_name(f".{destination.name}.absent"),
        "committing": destination.with_name(f".{destination.name}.committing"),
        "committed": destination.with_name(f".{destination.name}.committed"),
        "restore": destination.with_name(f".{destination.name}.restore"),
        "candidate": destination.with_name(f".{destination.name}.candidate"),
    }


def _transaction_payload(
    destination: Path,
    metadata: BidderCheckpointMeta,
    *,
    prior: str,
) -> dict[str, object]:
    return {
        "schema": "residual-checkpoint-transaction-v1",
        "destination_name": destination.name,
        "prior": prior,
        "checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "model_id": metadata.model_id,
        "status": metadata.status,
        **HASHES,
    }


def _write_marker(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def test_candidate_round_trip_preserves_all_members_and_metadata(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    metadata = _candidate(ensemble)
    path = tmp_path / "candidate.pt"

    save_checkpoint_atomic(path, ensemble, metadata)
    loaded, loaded_meta = load_checkpoint(path, **_expected_hashes())

    assert loaded_meta == metadata
    assert loaded_meta.schema == CHECKPOINT_SCHEMA
    assert loaded_meta.status == "candidate"
    assert loaded_meta.policy_id is None
    assert loaded_meta.calibration is None
    assert len(loaded.members) == 5
    for key, value in ensemble.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_model_and_policy_ids_are_content_addressed() -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    metadata = _candidate(ensemble)
    same = _candidate(ResidualQEnsemble(SEEDS))
    changed = ResidualQEnsemble(SEEDS)
    with torch.no_grad():
        next(changed.parameters()).add_(1.0)

    assert metadata.model_id == same.model_id
    assert metadata.model_id != _candidate(changed).model_id
    assert len(metadata.model_id) == 64

    calibration = CalibrationTuple(0.25, 0.75, 0.05, 0.8)
    promoted = promote_meta(metadata, calibration)
    assert promoted.status == "promoted"
    assert promoted.model_id == metadata.model_id
    assert promoted.policy_id is not None and len(promoted.policy_id) == 64
    assert promote_meta(metadata, calibration).policy_id == promoted.policy_id
    assert promote_meta(metadata, replace(calibration, epsilon=0.1)).policy_id != promoted.policy_id


def test_model_hash_has_a_versioned_framed_little_endian_contract() -> None:
    assert checkpoint_module.MODEL_ID_HASH_FORMAT == "sha256-framed-little-endian-v1"
    assert checkpoint_module._tensor_bytes_little_endian(
        torch.tensor([1.0, -2.5], dtype=torch.float32)
    ) == bytes.fromhex("0000803f000020c0")


def test_promoted_round_trip_and_policy_id_tamper_detection(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    promoted = promote_meta(_candidate(ensemble), CalibrationTuple(0.25, 0.75, 0.05, 0.8))
    path = tmp_path / "promoted.pt"
    save_checkpoint_atomic(path, ensemble, promoted)

    loaded, loaded_meta = load_checkpoint(
        path, require_promoted=True, **_expected_hashes()
    )

    assert loaded_meta == promoted
    for key, value in ensemble.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    artifact["metadata"]["policy_id"] = "a" * 64
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="policy_id"):
        load_checkpoint(path, require_promoted=True, **_expected_hashes())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda meta: replace(meta, schema="unknown"),
        lambda meta: replace(meta, status="promoted"),
        lambda meta: replace(meta, policy_id="a" * 64),
        lambda meta: replace(meta, calibration=CalibrationTuple(0.0, 0.0, 0.0, 1.0)),
        lambda meta: replace(meta, model_id="f" * 64),
    ],
)
def test_save_rejects_schema_status_and_identity_drift(tmp_path: Path, mutation) -> None:
    ensemble = ResidualQEnsemble(SEEDS)

    with pytest.raises(ValueError):
        save_checkpoint_atomic(tmp_path / "bad.pt", ensemble, mutation(_candidate(ensemble)))


def test_load_rejects_candidate_when_promoted_is_required(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    path = tmp_path / "candidate.pt"
    save_checkpoint_atomic(path, ensemble, _candidate(ensemble))

    with pytest.raises(ValueError, match="promoted"):
        load_checkpoint(path, require_promoted=True, **_expected_hashes())


@pytest.mark.parametrize("field", list(HASHES))
def test_load_fails_closed_on_frozen_hash_drift(tmp_path: Path, field: str) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    path = tmp_path / "candidate.pt"
    save_checkpoint_atomic(path, ensemble, _candidate(ensemble))
    expected = _expected_hashes()
    expected[f"expected_{field}"] = "9" * 64

    with pytest.raises(ValueError, match=field):
        load_checkpoint(path, **expected)


def test_load_rejects_wrong_member_count_or_dimensions(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    valid = tmp_path / "valid.pt"
    save_checkpoint_atomic(valid, ensemble, _candidate(ensemble))
    artifact = torch.load(valid, map_location="cpu", weights_only=True)

    missing_member = tmp_path / "missing.pt"
    artifact["state_dict"] = {
        key: value for key, value in artifact["state_dict"].items() if not key.startswith("members.4.")
    }
    torch.save(artifact, missing_member)
    with pytest.raises(ValueError):
        load_checkpoint(missing_member, **_expected_hashes())

    wrong_dimension = tmp_path / "dimension.pt"
    artifact = torch.load(valid, map_location="cpu", weights_only=True)
    artifact["state_dict"]["members.0.input_layer.weight"] = torch.zeros(256, 166)
    torch.save(artifact, wrong_dimension)
    with pytest.raises(ValueError):
        load_checkpoint(wrong_dimension, **_expected_hashes())


def test_load_rejects_nonfinite_parameters(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    valid = tmp_path / "valid.pt"
    save_checkpoint_atomic(valid, ensemble, _candidate(ensemble))
    artifact = torch.load(valid, map_location="cpu", weights_only=True)
    first = next(iter(artifact["state_dict"].values()))
    first.view(-1)[0] = float("nan")
    corrupt = tmp_path / "nonfinite.pt"
    torch.save(artifact, corrupt)

    with pytest.raises(ValueError, match="finite"):
        load_checkpoint(corrupt, **_expected_hashes())


def test_load_rejects_raw_serialized_dtype_drift_before_cast(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    valid = tmp_path / "valid.pt"
    save_checkpoint_atomic(valid, ensemble, _candidate(ensemble))
    artifact = torch.load(valid, map_location="cpu", weights_only=True)
    name = "members.0.input_layer.weight"
    artifact["state_dict"][name] = artifact["state_dict"][name].to(torch.float64)
    tampered = tmp_path / "dtype-drift.pt"
    torch.save(artifact, tampered)

    with pytest.raises(ValueError, match="dtype"):
        load_checkpoint(tampered, **_expected_hashes())


def test_load_rejects_raw_serialized_stride_drift_before_normalization(
    tmp_path: Path,
) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    valid = tmp_path / "valid.pt"
    save_checkpoint_atomic(valid, ensemble, _candidate(ensemble))
    artifact = torch.load(valid, map_location="cpu", weights_only=True)
    name = "members.0.input_layer.weight"
    original = artifact["state_dict"][name]
    backing = torch.empty((original.shape[0], original.shape[1] * 2), dtype=original.dtype)
    noncontiguous = backing[:, ::2]
    noncontiguous.copy_(original)
    assert noncontiguous.shape == original.shape
    assert noncontiguous.stride() != original.stride()
    artifact["state_dict"][name] = noncontiguous
    tampered = tmp_path / "stride-drift.pt"
    torch.save(artifact, tampered)

    with pytest.raises(ValueError, match="stride"):
        load_checkpoint(tampered, **_expected_hashes())


def test_load_rejects_raw_serialized_storage_offset_drift(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    valid = tmp_path / "valid.pt"
    save_checkpoint_atomic(valid, ensemble, _candidate(ensemble))
    artifact = torch.load(valid, map_location="cpu", weights_only=True)
    name = "members.0.input_layer.bias"
    original = artifact["state_dict"][name]
    backing = torch.empty(original.numel() + 1, dtype=original.dtype)
    offset = backing[1:]
    offset.copy_(original)
    assert offset.shape == original.shape
    assert offset.stride() == original.stride()
    assert offset.storage_offset() != original.storage_offset()
    artifact["state_dict"][name] = offset
    tampered = tmp_path / "offset-drift.pt"
    torch.save(artifact, tampered)

    with pytest.raises(ValueError, match="storage offset"):
        load_checkpoint(tampered, **_expected_hashes())


def test_atomic_save_validates_temporary_artifact_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"existing checkpoint")
    original = destination.read_bytes()

    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("validation failed")

    monkeypatch.setattr("residual_bidder.checkpoint._load_and_validate_artifact", reject)
    with pytest.raises(ValueError, match="validation failed"):
        save_checkpoint_atomic(destination, ensemble, _candidate(ensemble))

    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("destination_exists", [False, True])
def test_atomic_save_rolls_back_post_replace_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_exists: bool,
) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    old_bytes = b"exact prior checkpoint bytes"
    if destination_exists:
        destination.write_bytes(old_bytes)

    calls = 0
    # Both protocols durably record prior state first, then fsync the replacement.
    fail_on_call = 2

    def fail_once_after_replace(parent: Path) -> None:
        nonlocal calls
        assert parent == tmp_path
        calls += 1
        if calls == fail_on_call:
            raise OSError("injected directory fsync failure")

    monkeypatch.setattr(
        checkpoint_module,
        "_fsync_parent_directory",
        fail_once_after_replace,
        raising=False,
    )

    with pytest.raises(OSError, match="injected directory fsync failure"):
        save_checkpoint_atomic(destination, ensemble, _candidate(ensemble))

    assert destination.exists() is destination_exists
    if destination_exists:
        assert destination.read_bytes() == old_bytes
    assert not (tmp_path / f".{destination.name}.rollback").exists()
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_stale_uncommitted_backup_restores_exact_old_destination(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(destination, ensemble, _candidate(ensemble))
    paths = _transaction_paths(destination)
    old_bytes = b"exact old checkpoint bytes"
    paths["backup"].write_bytes(old_bytes)

    checkpoint_module._recover_checkpoint_transaction(destination)

    assert destination.read_bytes() == old_bytes
    assert not any(path.exists() for path in paths.values())


def test_stale_uncommitted_absence_marker_restores_absence(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    metadata = _candidate(ensemble)
    save_checkpoint_atomic(destination, ensemble, metadata)
    paths = _transaction_paths(destination)
    _write_marker(paths["absent"], _transaction_payload(destination, metadata, prior="absent"))

    checkpoint_module._recover_checkpoint_transaction(destination)

    assert not destination.exists()
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("prior", ["existing", "absent"])
def test_stale_committed_transaction_keeps_valid_new_destination(
    tmp_path: Path, prior: str
) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    metadata = _candidate(ensemble)
    save_checkpoint_atomic(destination, ensemble, metadata)
    expected_new = destination.read_bytes()
    paths = _transaction_paths(destination)
    payload = _transaction_payload(destination, metadata, prior=prior)
    if prior == "existing":
        paths["backup"].write_bytes(b"old checkpoint")
    else:
        _write_marker(paths["absent"], payload)
    _write_marker(paths["committed"], payload)

    checkpoint_module._recover_checkpoint_transaction(destination)

    assert destination.read_bytes() == expected_new
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("failure", ["link", "replace"])
def test_restore_link_or_replace_failure_never_consumes_only_old_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"new checkpoint")
    paths = _transaction_paths(destination)
    old_bytes = b"only durable old checkpoint"
    paths["backup"].write_bytes(old_bytes)
    original_link = checkpoint_module.os.link
    original_replace = checkpoint_module.os.replace

    def maybe_fail_link(source: Path, target: Path, **kwargs: object) -> None:
        if failure == "link" and Path(source) == paths["backup"]:
            raise OSError("injected restore link failure")
        original_link(source, target, **kwargs)

    def maybe_fail_replace(source: Path, target: Path) -> None:
        if failure == "replace" and Path(source) == paths["restore"]:
            raise OSError("injected restore replace failure")
        original_replace(source, target)

    monkeypatch.setattr(checkpoint_module.os, "link", maybe_fail_link)
    monkeypatch.setattr(checkpoint_module.os, "replace", maybe_fail_replace)

    with pytest.raises(OSError, match=f"restore {failure} failure"):
        checkpoint_module._recover_checkpoint_transaction(destination)

    assert paths["backup"].read_bytes() == old_bytes
    if paths["restore"].exists():
        assert paths["restore"].read_bytes() == old_bytes


@pytest.mark.parametrize("prior", ["existing", "absent"])
def test_recovery_fsync_failure_leaves_idempotent_prior_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior: str,
) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"new checkpoint")
    paths = _transaction_paths(destination)
    old_bytes = b"old checkpoint"
    if prior == "existing":
        paths["backup"].write_bytes(old_bytes)
    else:
        _write_marker(
            paths["absent"],
            {
                "schema": "residual-checkpoint-transaction-v1",
                "destination_name": destination.name,
                "prior": "absent",
                "checkpoint_sha256": "0" * 64,
                "model_id": "0" * 64,
                "status": "candidate",
                **HASHES,
            },
        )
    original_fsync = checkpoint_module._fsync_parent_directory
    calls = 0

    def fail_once(parent: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected recovery fsync failure")
        original_fsync(parent)

    monkeypatch.setattr(checkpoint_module, "_fsync_parent_directory", fail_once)
    with pytest.raises(OSError, match="recovery fsync failure"):
        checkpoint_module._recover_checkpoint_transaction(destination)

    evidence = paths["backup"] if prior == "existing" else paths["absent"]
    assert evidence.exists()

    monkeypatch.setattr(checkpoint_module, "_fsync_parent_directory", original_fsync)
    checkpoint_module._recover_checkpoint_transaction(destination)
    assert destination.exists() is (prior == "existing")
    if prior == "existing":
        assert destination.read_bytes() == old_bytes
    assert not any(path.exists() for path in paths.values())


def test_committed_cleanup_is_directory_fsynced_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    metadata = _candidate(ensemble)
    save_checkpoint_atomic(destination, ensemble, metadata)
    paths = _transaction_paths(destination)
    payload = _transaction_payload(destination, metadata, prior="existing")
    paths["backup"].write_bytes(b"old checkpoint")
    _write_marker(paths["committed"], payload)
    original_fsync = checkpoint_module._fsync_parent_directory
    calls = 0

    def fail_second_cleanup_fsync(parent: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected final cleanup fsync failure")
        original_fsync(parent)

    monkeypatch.setattr(
        checkpoint_module, "_fsync_parent_directory", fail_second_cleanup_fsync
    )
    with pytest.raises(
        checkpoint_module.CheckpointCommittedCleanupError,
        match="committed",
    ):
        checkpoint_module._recover_checkpoint_transaction(destination)
    assert calls == 2

    monkeypatch.setattr(checkpoint_module, "_fsync_parent_directory", original_fsync)
    checkpoint_module._recover_checkpoint_transaction(destination)
    assert destination.exists()
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("corruption", ["both-priors", "corrupt-commit"])
def test_contradictory_or_corrupt_transaction_markers_fail_closed(
    tmp_path: Path, corruption: str
) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"new checkpoint")
    paths = _transaction_paths(destination)
    paths["backup"].write_bytes(b"old checkpoint")
    if corruption == "both-priors":
        paths["absent"].write_text("{}\n", encoding="ascii")
    else:
        paths["committed"].write_text("not-json\n", encoding="ascii")
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}

    with pytest.raises(ValueError, match="transaction"):
        checkpoint_module._recover_checkpoint_transaction(destination)

    assert destination.read_bytes() == b"new checkpoint"
    assert {name: path.read_bytes() for name, path in paths.items() if path.exists()} == before


def test_mismatched_absence_and_committed_markers_fail_closed(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    destination = tmp_path / "checkpoint.pt"
    metadata = _candidate(ensemble)
    save_checkpoint_atomic(destination, ensemble, metadata)
    paths = _transaction_paths(destination)
    committed = _transaction_payload(destination, metadata, prior="absent")
    absence = dict(committed)
    absence["model_id"] = "f" * 64
    _write_marker(paths["absent"], absence)
    _write_marker(paths["committed"], committed)
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}

    with pytest.raises(ValueError, match="transaction"):
        checkpoint_module._recover_checkpoint_transaction(destination)

    assert {name: path.read_bytes() for name, path in paths.items() if path.exists()} == before


def test_corrupt_in_progress_commit_marker_fails_closed(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"new checkpoint")
    paths = _transaction_paths(destination)
    paths["backup"].write_bytes(b"old checkpoint")
    paths["committing"].write_text("partial-json", encoding="ascii")
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}

    with pytest.raises(ValueError, match="transaction"):
        checkpoint_module._recover_checkpoint_transaction(destination)

    assert destination.read_bytes() == b"new checkpoint"
    assert {name: path.read_bytes() for name, path in paths.items() if path.exists()} == before


def test_checkpoint_artifact_contains_only_safe_primitive_metadata_and_tensors(tmp_path: Path) -> None:
    ensemble = ResidualQEnsemble(SEEDS)
    path = tmp_path / "candidate.pt"
    save_checkpoint_atomic(path, ensemble, _candidate(ensemble))

    artifact = torch.load(path, map_location="cpu", weights_only=True)

    assert set(artifact) == {"metadata", "state_dict"}
    assert isinstance(artifact["metadata"], dict)
    assert all(isinstance(key, str) for key in artifact["metadata"])
    assert all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in artifact["state_dict"].items()
    )
