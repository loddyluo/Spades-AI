"""Unit tests for the new full-information MLP training utilities.

Inputs:
- `FullInfoValueMLP` expects input feature vectors of length D.
- `split_samples_by_bid` expects samples with `state_summary` and `current_player`.
- `prepare_value_arrays` expects samples with `feature` and `value_view`.

Outputs:
- Validates model output shapes.
- Validates bid-based sample splitting and value scaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_mlp.model import FullInfoValueMLP
from new_mlp.train import prepare_value_arrays, split_samples_by_bid


def test_full_info_value_mlp_forward_shape() -> None:
    """Check that the value model outputs a (1,1) tensor for single inputs.

    Input:
    - Single feature vector with length 1385.

    Output:
    - Asserts predict returns a scalar float.
    """
    model = FullInfoValueMLP(input_dim=1385, hidden_dims=[32])
    features = np.zeros(1385, dtype=np.float32)
    pred = model.predict(features)
    assert isinstance(pred, float)


def test_split_samples_by_bid_and_value_scaling() -> None:
    """Verify bid-based splitting and value scaling logic.

    Input:
    - Two samples: one nil bid (treated as 0), one bid_2.

    Output:
    - Asserts correct split sizes and scaled value targets.
    """
    samples = [
        {
            "feature": np.zeros(1385, dtype=np.float32),
            "value_view": 25.0,
            "state_summary": {"bids": ["nil", "bid_1", "bid_2", "bid_3"]},
            "current_player": 0,
        },
        {
            "feature": np.ones(1385, dtype=np.float32),
            "value_view": -50.0,
            "state_summary": {"bids": ["bid_2", "bid_1", "bid_2", "bid_3"]},
            "current_player": 0,
        },
    ]

    bid0_samples, bidpos_samples = split_samples_by_bid(samples)
    assert len(bid0_samples) == 1
    assert len(bidpos_samples) == 1

    arrays = prepare_value_arrays(bid0_samples, value_scale=25.0)
    assert arrays["features"].shape == (1, 1385)
    assert arrays["value_targets"].shape == (1, 1)
    assert np.isclose(arrays["value_targets"][0, 0], 1.0)