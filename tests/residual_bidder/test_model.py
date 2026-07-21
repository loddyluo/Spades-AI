from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from residual_bidder.actions import BidAction, LocalNeighborhood
from residual_bidder.model import (
    ResidualQEnsemble,
    ResidualQMember,
    build_residual_input,
    mask_invalid_alternatives,
    masked_mse_loss,
)
from residual_bidder.nsfp import NSFPObservation


def _observation(center: BidAction) -> NSFPObservation:
    encoded = torch.linspace(-3.0, 3.0, 149, dtype=torch.float32)
    scores = torch.linspace(-7.0, 6.0, 14, dtype=torch.float32)
    scores[int(center)] = 20.0
    return NSFPObservation(
        encoded_149=encoded,
        raw_logits_16=torch.zeros(16),
        legal_scores_14=scores,
        center=center,
    )


@pytest.mark.parametrize(
    ("center", "expected_neighborhood", "expected_mask"),
    [
        (
            BidAction.NIL,
            LocalNeighborhood(BidAction.NIL, None, BidAction.BID_1),
            [0.0, 1.0],
        ),
        (
            BidAction.BID_7,
            LocalNeighborhood(BidAction.BID_7, BidAction.BID_6, BidAction.BID_8),
            [1.0, 1.0],
        ),
        (
            BidAction.BID_13,
            LocalNeighborhood(BidAction.BID_13, BidAction.BID_12, None),
            [1.0, 0.0],
        ),
    ],
)
def test_residual_input_has_exact_167_value_contract(
    center: BidAction,
    expected_neighborhood: LocalNeighborhood,
    expected_mask: list[float],
) -> None:
    observation = _observation(center)

    result = build_residual_input(observation)

    assert result.values.shape == (167,)
    assert result.values.dtype == observation.encoded_149.dtype
    assert torch.equal(result.values[:149], observation.encoded_149)
    assert torch.equal(
        result.values[149:163],
        torch.nn.functional.one_hot(torch.tensor(int(center)), num_classes=14).to(torch.float32),
    )
    expected_margins = [
        0.0
        if expected_neighborhood.lower is None
        else float(
            (observation.legal_scores_14[int(center)]
             - observation.legal_scores_14[int(expected_neighborhood.lower)])
            / 13.47
        ),
        0.0
        if expected_neighborhood.upper is None
        else float(
            (observation.legal_scores_14[int(center)]
             - observation.legal_scores_14[int(expected_neighborhood.upper)])
            / 13.47
        ),
    ]
    assert result.values[163:165].tolist() == pytest.approx(expected_margins)
    assert result.values[165:167].tolist() == expected_mask
    assert result.alternative_mask.tolist() == expected_mask
    assert result.neighborhood == expected_neighborhood


def test_residual_input_has_only_the_public_observation_input_route() -> None:
    signature = inspect.signature(build_residual_input)
    assert list(signature.parameters) == ["obs", "margin_divisor"]
    assert signature.parameters["obs"].annotation in {"NSFPObservation", NSFPObservation}

    with pytest.raises(TypeError, match="NSFPObservation"):
        build_residual_input(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_residual_input(_observation(BidAction.BID_4), opponent_hands=[])  # type: ignore[call-arg]


@pytest.mark.parametrize("bad_divisor", [0.0, -1.0, float("nan"), float("inf")])
def test_residual_input_rejects_invalid_margin_divisor(bad_divisor: float) -> None:
    with pytest.raises(ValueError):
        build_residual_input(_observation(BidAction.BID_4), bad_divisor)


def test_member_has_the_exact_dropout_free_architecture() -> None:
    member = ResidualQMember()
    modules = list(member.modules())

    assert [(layer.in_features, layer.out_features) for layer in modules if isinstance(layer, nn.Linear)] == [
        (167, 256),
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 128),
        (128, 2),
    ]
    assert [layer.normalized_shape for layer in modules if isinstance(layer, nn.LayerNorm)] == [
        (256,),
        (256,),
        (256,),
        (128,),
    ]
    assert len([layer for layer in modules if isinstance(layer, nn.SiLU)]) == 6
    assert not any(isinstance(layer, nn.Dropout) for layer in modules)


def test_member_and_ensemble_shapes_are_exact_and_eval_is_deterministic() -> None:
    member = ResidualQMember().eval()
    ensemble = ResidualQEnsemble((11, 12, 13, 14, 15)).eval()
    single = torch.randn(167)
    batch = torch.randn(3, 4, 167)

    assert member(single).shape == (2,)
    assert member(batch).shape == (3, 4, 2)
    assert ensemble(single).shape == (5, 2)
    assert ensemble(batch).shape == (5, 3, 4, 2)
    assert torch.equal(ensemble(batch), ensemble(batch))


def test_members_are_seeded_independently_without_shared_parameter_storage() -> None:
    seeds = (101, 202, 303, 404, 505)
    ensemble = ResidualQEnsemble(seeds)
    changed_first_seed = ResidualQEnsemble((999, *seeds[1:]))

    pointers = [parameter.data_ptr() for member in ensemble.members for parameter in member.parameters()]
    assert len(ensemble.members) == 5
    assert len(pointers) == len(set(pointers))
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            ensemble.members[0].parameters(),
            changed_first_seed.members[0].parameters(),
            strict=True,
        )
    )
    for index in range(1, 5):
        for left, right in zip(
            ensemble.members[index].parameters(),
            changed_first_seed.members[index].parameters(),
            strict=True,
        ):
            assert torch.equal(left, right)


def test_ensemble_construction_does_not_change_the_callers_rng_stream() -> None:
    torch.manual_seed(77)
    before = torch.random.get_rng_state()
    ResidualQEnsemble((1, 2, 3, 4, 5))
    after = torch.random.get_rng_state()

    assert torch.equal(after, before)


def test_mean_std_uses_population_standard_deviation() -> None:
    ensemble = ResidualQEnsemble((1, 2, 3, 4, 5)).eval()
    inputs = torch.randn(2, 167)

    outputs = ensemble(inputs)
    mean, std = ensemble.mean_std(inputs)

    assert torch.equal(mean, outputs.mean(dim=0))
    assert torch.equal(std, outputs.std(dim=0, unbiased=False))


def test_lower_and_upper_slots_are_masked_without_remapping() -> None:
    values = torch.tensor([[7.0, 11.0], [13.0, 17.0]])
    masks = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    inference = mask_invalid_alternatives(values, masks)

    assert inference[0, 0] == -torch.inf
    assert inference[0, 1] == 11.0
    assert inference[1, 0] == 13.0
    assert inference[1, 1] == -torch.inf
    assert masked_mse_loss(values, torch.zeros_like(values), masks) == pytest.approx(
        (11.0**2 + 13.0**2) / 2
    )


def test_masked_mse_never_evaluates_nonfinite_missing_elements() -> None:
    predictions = torch.tensor(
        [[2.0, float("nan")], [float("inf"), 3.0]], requires_grad=True
    )
    targets = torch.tensor([[1.0, float("inf")], [float("nan"), 1.0]])
    masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = masked_mse_loss(predictions, targets, masks)
    loss.backward()

    assert loss.item() == pytest.approx(2.5)
    assert predictions.grad is not None
    assert torch.equal(predictions.grad, torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    assert bool(torch.isfinite(predictions.grad).all().item())
