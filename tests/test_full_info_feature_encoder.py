"""Regression tests for the full-information feature encoder.

Inputs:
- `FullInfoSpadesFeatureEncoder.encode(state, player_id)` expects a `GameState`
	with all players' hands populated and a perspective player id.
- `FullInfoSpadesFeatureEncoder.encode_sections(state, player_id)` returns the
	hand section that now includes opponent hands.

Outputs:
- Asserts that the hand section encodes the current player's hand and the
	other three players' remaining cards in the documented one-hot layout.
- Asserts that the full feature vector length matches `encoder.total_dim`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(REPO_ROOT))

from data.training_data import build_state_with_remaining_cards
from trick_taking.utils.feature_encoder import FullInfoSpadesFeatureEncoder


def test_full_info_encoder_hand_section_contains_opponent_hands() -> None:
		"""Verify opponent hand encoding in the hand section.

		Input:
		- A deterministic state built with `build_state_with_remaining_cards`.

		Output:
		- Asserts that the hand section contains the exact one-hot cards for the
			current player and each opponent (ordered by seat id, excluding self).
		"""
		state = build_state_with_remaining_cards(24, seed=1234)
		player_id = int(state.turn)
		encoder = FullInfoSpadesFeatureEncoder()

		sections = encoder.encode_sections(state, player_id)
		hand_section = sections["hand"]

		expected_own = np.zeros(52, dtype=np.float32)
		for card in state.hands[player_id]:
				expected_own[card.card_id] = 1.0

		expected_opp = np.zeros(3 * 52, dtype=np.float32)
		opp_ids = [pid for pid in range(4) if pid != player_id]
		for idx, pid in enumerate(opp_ids):
				for card in state.hands[pid]:
						expected_opp[idx * 52 + card.card_id] = 1.0

		assert hand_section.shape[0] == encoder.DIM_HAND
		np.testing.assert_array_equal(hand_section[:52], expected_own)
		np.testing.assert_array_equal(hand_section[52:52 + 3 * 52], expected_opp)


def test_full_info_encoder_total_dim_matches_output() -> None:
    """Verify the full vector length equals the encoder's declared total_dim.

    Input:
    - A deterministic state built with `build_state_with_remaining_cards`.

    Output:
    - Asserts that the encoded feature length equals `encoder.total_dim`.
    """
    state = build_state_with_remaining_cards(24, seed=777)
    encoder = FullInfoSpadesFeatureEncoder()
    features = encoder.encode(state, int(state.turn))
    assert features.shape[0] == encoder.total_dim