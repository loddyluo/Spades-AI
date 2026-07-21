"""Fit one experimental residual-Q ensemble from minimal hybrid NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from residual_bidder.checkpoint import build_candidate_meta, save_checkpoint_atomic
from residual_bidder.config import BidderConfig, ConfigError, canonical_sha256
from residual_bidder.hybrid import concatenate_hybrid_arrays, load_hybrid_npz
from residual_bidder.training import fit_residual_ensemble


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _play_pipeline_sha256(config: BidderConfig) -> str:
    source_manifest = [
        [source, _sha256(Path(source))] for source in config.play.source_manifest
    ]
    return canonical_sha256(
        {
            "play_config_sha256": config.play.config_sha256,
            "source_manifest": source_manifest,
        }
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def train_to_checkpoint(
    config: BidderConfig,
    *,
    train_path: Path | Sequence[Path],
    validation_path: Path | Sequence[Path],
    output: Path,
    device: torch.device,
    max_epochs: int | None = None,
) -> dict[str, object]:
    train_paths = [train_path] if isinstance(train_path, Path) else list(train_path)
    validation_paths = (
        [validation_path]
        if isinstance(validation_path, Path)
        else list(validation_path)
    )
    if not train_paths or not validation_paths:
        raise ValueError("training and validation paths must be nonempty")
    train = concatenate_hybrid_arrays(
        [load_hybrid_npz(path) for path in train_paths]
    )
    validation = concatenate_hybrid_arrays(
        [load_hybrid_npz(path) for path in validation_paths]
    )
    member_seeds = tuple(config.model.init_seeds)
    if len(member_seeds) != 5:
        raise ValueError("minimal trainer requires exactly five member seeds")
    resolved_epochs = config.training.max_epochs if max_epochs is None else max_epochs

    started = time.perf_counter()
    result = fit_residual_ensemble(
        train,
        validation,
        member_init_seeds=member_seeds,  # type: ignore[arg-type]
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_epochs=resolved_epochs,
        patience=config.training.early_stop_patience,
        gradient_norm_clip=config.training.gradient_norm_clip,
        device=device,
        training_seed=config.policy.policy_seed,
    )
    training_seconds = time.perf_counter() - started

    dataset_sha256 = canonical_sha256(
        {
            "train_sha256s": [_sha256(path) for path in train_paths],
            "validation_sha256s": [_sha256(path) for path in validation_paths],
        }
    )
    metadata = build_candidate_meta(
        result.ensemble,
        iteration=0,
        nsfp_sha256=config.nsfp.sha256,
        play_pipeline_sha256=_play_pipeline_sha256(config),
        config_sha256=config.sha256(),
        dataset_manifest_sha256=dataset_sha256,
        member_init_seeds=member_seeds,  # type: ignore[arg-type]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint_atomic(output, result.ensemble, metadata)
    return {
        "ok": True,
        "schema": "minimal-residual-training-v1",
        "device": str(device),
        "train_files": len(train_paths),
        "validation_files": len(validation_paths),
        "train_rows": int(train.features.shape[0]),
        "validation_rows": int(validation.features.shape[0]),
        "epochs_ran": result.epochs_ran,
        "best_epoch": result.best_epoch,
        "training_seconds": training_seconds,
        "examples_per_second": (
            int(train.features.shape[0]) * result.epochs_ran / training_seconds
        ),
        "final_training_loss": result.final_training_loss,
        "best_validation_mse": result.best_validation_mse,
        "zero_validation_mse": result.zero_validation_mse,
        "validation_sign_accuracy": result.validation_sign_accuracy,
        "beats_zero_validation_mse": (
            result.best_validation_mse < result.zero_validation_mse
        ),
        "model_id": metadata.model_id,
        "dataset_sha256": dataset_sha256,
        "checkpoint": str(output),
        "checkpoint_bytes": output.stat().st_size,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_bidder/base.yaml"),
    )
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = train_to_checkpoint(
            BidderConfig.load(arguments.config),
            train_path=arguments.train,
            validation_path=arguments.validation,
            output=arguments.output,
            device=_device(arguments.device),
            max_epochs=arguments.max_epochs,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
