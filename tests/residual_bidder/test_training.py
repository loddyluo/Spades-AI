from __future__ import annotations

import numpy as np
import pytest
import torch

from residual_bidder.hybrid import HybridArrays
from residual_bidder.training import bootstrap_multiplicities, fit_residual_ensemble


SEEDS = (1701, 1702, 1703, 1704, 1705)


def _arrays(deal_numbers: list[int]) -> HybridArrays:
    rows = len(deal_numbers) * 4
    rng = np.random.default_rng(sum(deal_numbers))
    features = rng.normal(size=(rows, 167)).astype(np.float32)
    targets = np.stack((features[:, 0] * 0.1, features[:, 1] * -0.1), axis=1)
    return HybridArrays(
        features=features,
        targets=targets.astype(np.float32),
        masks=np.ones((rows, 2), dtype=np.float32),
        centers=np.full(rows, 5, dtype=np.int8),
        baseline_margins=np.zeros(rows, dtype=np.float64),
        shuffle_seeds=np.repeat(np.asarray(deal_numbers, dtype=np.int64), 4),
        room_ids=np.tile(np.asarray([0, 1, 0, 1], dtype=np.int8), len(deal_numbers)),
        physical_seats=np.tile(np.arange(4, dtype=np.int8), len(deal_numbers)),
        bid_indices=np.tile(np.arange(4, dtype=np.int8), len(deal_numbers)),
        deal_ids=np.repeat(np.asarray([f"deal-{number}" for number in deal_numbers]), 4),
    )


def test_bootstrap_multiplicities_are_deal_grouped_and_reproducible() -> None:
    deal_ids = np.asarray(["a"] * 4 + ["b"] * 4 + ["c"] * 4)

    first = bootstrap_multiplicities(deal_ids, SEEDS)
    second = bootstrap_multiplicities(deal_ids, SEEDS)

    assert torch.equal(first, second)
    assert first.shape == (5, 12)
    for member in range(5):
        for start in (0, 4, 8):
            assert torch.unique(first[member, start : start + 4]).numel() == 1
        assert int(first[member, ::4].sum().item()) == 3


def test_fit_rejects_train_validation_deal_overlap() -> None:
    data = _arrays([10, 11])

    with pytest.raises(ValueError, match="disjoint"):
        fit_residual_ensemble(
            data,
            data,
            member_init_seeds=SEEDS,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_epochs=1,
            patience=1,
            gradient_norm_clip=1.0,
            device=torch.device("cpu"),
        )


def test_minimal_fit_returns_a_finite_best_model_and_metrics() -> None:
    train = _arrays([20, 21])
    validation = _arrays([30])

    result = fit_residual_ensemble(
        train,
        validation,
        member_init_seeds=SEEDS,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=2,
        patience=2,
        gradient_norm_clip=1.0,
        device=torch.device("cpu"),
    )

    assert 1 <= result.epochs_ran <= 2
    assert 1 <= result.best_epoch <= result.epochs_ran
    assert np.isfinite(result.best_validation_mse)
    assert np.isfinite(result.zero_validation_mse)
    assert 0.0 <= result.validation_sign_accuracy <= 1.0
    assert result.ensemble.training is False
    assert all(torch.isfinite(parameter).all() for parameter in result.ensemble.parameters())
