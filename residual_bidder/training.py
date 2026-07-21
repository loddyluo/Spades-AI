"""Minimal masked-MSE fitting for the five-member residual-Q ensemble."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from residual_bidder.hybrid import HybridArrays
from residual_bidder.model import ENSEMBLE_MEMBERS, ResidualQEnsemble


@dataclass(frozen=True)
class TrainingResult:
    ensemble: ResidualQEnsemble
    epochs_ran: int
    best_epoch: int
    final_training_loss: float
    best_validation_mse: float
    zero_validation_mse: float
    validation_sign_accuracy: float


def _validate_arrays(data: HybridArrays, name: str) -> None:
    if not isinstance(data, HybridArrays):
        raise TypeError(f"{name} must be HybridArrays")
    rows = data.features.shape[0]
    if rows <= 0 or data.features.shape != (rows, 167):
        raise ValueError(f"{name} features must have shape (N, 167) with N > 0")
    if data.targets.shape != (rows, 2) or data.masks.shape != (rows, 2):
        raise ValueError(f"{name} targets and masks must have shape (N, 2)")
    if not np.isfinite(data.features).all() or not np.isfinite(data.targets).all():
        raise ValueError(f"{name} features and targets must be finite")
    if not np.isin(data.masks, (0.0, 1.0)).all() or not data.masks.any(axis=1).all():
        raise ValueError(f"{name} masks must contain a legal alternative in every row")
    if data.deal_ids.shape != (rows,):
        raise ValueError(f"{name} deal_ids must have shape (N,)")


def bootstrap_multiplicities(
    deal_ids: np.ndarray,
    member_init_seeds: tuple[int, int, int, int, int],
) -> torch.Tensor:
    """Draw one deterministic with-replacement deal bootstrap per member."""

    if not isinstance(deal_ids, np.ndarray) or deal_ids.ndim != 1 or not len(deal_ids):
        raise ValueError("deal_ids must be a nonempty one-dimensional NumPy array")
    if len(member_init_seeds) != ENSEMBLE_MEMBERS:
        raise ValueError("exactly five member seeds are required")
    unique_deals, inverse = np.unique(deal_ids.astype(str), return_inverse=True)
    deal_count = len(unique_deals)
    rows: list[np.ndarray] = []
    for seed in member_init_seeds:
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, deal_count, size=deal_count)
        counts = np.bincount(sampled, minlength=deal_count).astype(np.float32)
        rows.append(counts[inverse])
    return torch.from_numpy(np.stack(rows, axis=0))


def _masked_mse(predictions: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    legal = masks.to(dtype=torch.bool)
    return (predictions.masked_select(legal) - targets.masked_select(legal)).square().mean()


def _predict_mean(
    ensemble: ResidualQEnsemble,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    ensemble.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            predictions.append(ensemble(batch).mean(dim=0).cpu())
    return torch.cat(predictions, dim=0)


def _evaluation_metrics(
    ensemble: ResidualQEnsemble,
    validation: HybridArrays,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    features = torch.from_numpy(validation.features.astype(np.float32, copy=False))
    targets = torch.from_numpy(validation.targets.astype(np.float32, copy=False))
    masks = torch.from_numpy(validation.masks.astype(np.float32, copy=False))
    predictions = _predict_mean(
        ensemble, features, batch_size=batch_size, device=device
    )
    mse = float(_masked_mse(predictions, targets, masks).item())
    signed = masks.to(dtype=torch.bool) & (targets != 0)
    if bool(signed.any().item()):
        sign_accuracy = float(
            ((predictions > 0) == (targets > 0)).masked_select(signed).float().mean().item()
        )
    else:
        sign_accuracy = 1.0
    return mse, sign_accuracy


def fit_residual_ensemble(
    train: HybridArrays,
    validation: HybridArrays,
    *,
    member_init_seeds: tuple[int, int, int, int, int],
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gradient_norm_clip: float,
    device: torch.device,
    training_seed: int = 20260721,
) -> TrainingResult:
    """Fit one experimental ensemble and retain its best validation epoch."""

    _validate_arrays(train, "train")
    _validate_arrays(validation, "validation")
    if set(train.deal_ids.astype(str)) & set(validation.deal_ids.astype(str)):
        raise ValueError("training and validation deal IDs must be disjoint")
    for name, value in (
        ("batch_size", batch_size),
        ("max_epochs", max_epochs),
        ("patience", patience),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value, allow_zero in (
        ("learning_rate", learning_rate, False),
        ("weight_decay", weight_decay, True),
        ("gradient_norm_clip", gradient_norm_clip, False),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} is outside its supported range")
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")

    features = torch.from_numpy(train.features.astype(np.float32, copy=False))
    targets = torch.from_numpy(train.targets.astype(np.float32, copy=False))
    masks = torch.from_numpy(train.masks.astype(np.float32, copy=False))
    bootstrap = bootstrap_multiplicities(train.deal_ids, member_init_seeds)

    ensemble = ResidualQEnsemble(member_init_seeds).to(device)
    optimizer = torch.optim.AdamW(
        ensemble.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(training_seed)

    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    final_training_loss = math.inf
    epochs_ran = 0

    for epoch in range(1, max_epochs + 1):
        ensemble.train()
        order = torch.randperm(len(features), generator=order_generator)
        weighted_loss_sum = 0.0
        batch_count = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_targets = targets[indices].to(device)
            batch_masks = masks[indices].to(device)
            batch_bootstrap = bootstrap[:, indices].to(device)
            predictions = ensemble(batch_features)
            weights = batch_bootstrap.unsqueeze(-1) * batch_masks.unsqueeze(0)
            denominator = weights.sum()
            if denominator.item() <= 0:
                continue
            loss = ((predictions - batch_targets.unsqueeze(0)).square() * weights).sum() / denominator
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("training loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), float(gradient_norm_clip))
            optimizer.step()
            weighted_loss_sum += float(loss.detach().cpu().item())
            batch_count += 1
        if batch_count == 0:
            raise ValueError("no training batch contained positive bootstrap weight")
        final_training_loss = weighted_loss_sum / batch_count
        validation_mse, _ = _evaluation_metrics(
            ensemble,
            validation,
            batch_size=batch_size,
            device=device,
        )
        epochs_ran = epoch
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in ensemble.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None or best_epoch == 0:
        raise AssertionError("training did not produce a best validation state")
    ensemble.load_state_dict(best_state, strict=True)
    ensemble.eval()
    best_validation, sign_accuracy = _evaluation_metrics(
        ensemble,
        validation,
        batch_size=batch_size,
        device=device,
    )
    validation_targets = torch.from_numpy(
        validation.targets.astype(np.float32, copy=False)
    )
    validation_masks = torch.from_numpy(
        validation.masks.astype(np.float32, copy=False)
    )
    zero_mse = float(
        _masked_mse(
            torch.zeros_like(validation_targets), validation_targets, validation_masks
        ).item()
    )
    return TrainingResult(
        ensemble=ensemble,
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        final_training_loss=final_training_loss,
        best_validation_mse=best_validation,
        zero_validation_mse=zero_mse,
        validation_sign_accuracy=sign_accuracy,
    )
