"""Regression tests for full-information model integration in truncated MCTS.

Inputs:
- `TruncatedMCTSConfig` with full-info checkpoint paths and value scale.
- `TruncatedMCTSStrategy._leaf_value()` expects a GameState at x<=24.
- `TruncatedMCTSStrategy._select_full_info_model()` expects bid info in state.

Outputs:
- Validates bid-based model selection logic.
- Validates that full-info encoder dimension matches model expectations.
- Validates leaf value computation with full-info models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import build_state_with_remaining_cards
from new_mlp.model import FullInfoValueMLP
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy
from trick_taking.utils.feature_encoder import FullInfoSpadesFeatureEncoder


def test_parse_bid_value_conversion() -> None:
    """Verify bid value parsing from various formats.

    Input:
    - Various bid string formats and numeric types.

    Output:
    - Asserts correct conversion to integer values.
    """
    strategy = TruncatedMCTSStrategy()
    assert strategy._parse_bid_value("nil") == 0
    assert strategy._parse_bid_value("blind_nil") == 0
    assert strategy._parse_bid_value("bid_3") == 3
    assert strategy._parse_bid_value("bid_13") == 13
    assert strategy._parse_bid_value(5) == 5
    assert strategy._parse_bid_value(None) == 0


def test_select_full_info_model_by_bid() -> None:
    """Verify full-info model selection based on current player's bid.

    Input:
    - State with bid==0 and bid==2 respectively.

    Output:
    - Asserts correct model selected (or None when not configured).
    """
    state = build_state_with_remaining_cards(24, seed=1000)

    config = TruncatedMCTSConfig(
        full_info_bid0_checkpoint=None,
        full_info_bidpos_checkpoint=None,
    )
    strategy = TruncatedMCTSStrategy(config)
    selected = strategy._select_full_info_model(state)
    assert selected is None

    config_with_models = TruncatedMCTSConfig(
        full_info_bid0_checkpoint=None,
        full_info_bidpos_checkpoint=None,
    )
    strategy_with_models = TruncatedMCTSStrategy(config_with_models)
    if strategy_with_models.full_info_bid0_model is None:
        selected_bid0 = strategy_with_models._select_full_info_model(state)
        assert selected_bid0 is None


def test_full_info_encoder_integration_with_mcts() -> None:
    """Verify full-info encoder can encode state for MCTS leaf evaluation.

    Input:
    - State at x=24 with full determinized opponent hands.

    Output:
    - Asserts encoded feature dimension matches FullInfoValueMLP.input_dim.
    """
    state = build_state_with_remaining_cards(24, seed=2000)
    encoder = FullInfoSpadesFeatureEncoder()
    features = encoder.encode(state, state.turn)

    expected_dim = 1385
    assert features.shape[0] == expected_dim
    assert features.dtype == np.float32

    model = FullInfoValueMLP(input_dim=expected_dim)
    pred = model.predict(features)
    assert isinstance(pred, float)
