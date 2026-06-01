"""Feature privacy and value-scaling regression tests for the MLP data path.

Inputs:
- `prepare_multi_head_arrays(samples)` expects a list of sample dictionaries with at least
  `feature`, `value_view`, `action_ids`, `action_q_values`, and `best_action_id` fields.
- `SpadesFeatureEncoder.encode(state, player_id)` expects a fully constructed
  `GameState` and the perspective player id.

Outputs:
- The scaling test checks that the prepared `value_targets` are `value_view / 25.0`.
- The privacy test checks that feature vectors stay identical when only hidden opponent
  card identities are permuted while all public information is preserved.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import build_state_with_remaining_cards
from mlp.training_utils import prepare_multi_head_arrays
from trick_taking.card import cards_to_bitset
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def _permute_hidden_opponent_cards(state, player_id: int, seed: int) -> object:
    """Return a deep-copied state with opponent hidden cards reshuffled.

    Inputs:
    - state: a fully populated `GameState` instance.
    - player_id: the perspective player whose own hand and public view must stay unchanged.
    - seed: RNG seed for deterministic shuffling.

    Output:
    - A deep-copied state where only the identities of cards held by players other than
      `player_id` are permuted. Public information and hand sizes remain unchanged.
    """
    shuffled_state = copy.deepcopy(state)
    rng = random.Random(seed)

    hidden_cards = []
    for pid, hand in enumerate(shuffled_state.hands):
        if pid == player_id:
            continue
        hidden_cards.extend(hand)

    rng.shuffle(hidden_cards)

    offset = 0
    for pid, hand in enumerate(shuffled_state.hands):
        if pid == player_id:
            continue
        count = len(hand)
        shuffled_state.hands[pid] = hidden_cards[offset : offset + count]
        offset += count

    shuffled_state.hand_bitsets = [cards_to_bitset(hand) for hand in shuffled_state.hands]
    return shuffled_state


def test_prepare_multi_head_arrays_scales_value_view_by_25() -> None:
    """Verify that the training target is built from `value_view / 25.0`.

    Inputs:
    - A single synthetic sample with `value_view=50.0` and an empty legal-action set.

    Output:
    - Asserts that the prepared value target becomes `2.0`.
    """
    samples = [
        {
            "feature": torch.zeros(1229, dtype=torch.float32),
            "value_view": torch.tensor(50.0, dtype=torch.float32),
            "action_ids": torch.tensor([], dtype=torch.int64),
            "action_q_values": torch.tensor([], dtype=torch.float32),
            "best_action_id": -1,
        }
    ]

    arrays = prepare_multi_head_arrays(samples)
    assert arrays["value_targets"].shape == (1, 1)
    assert np.isclose(arrays["value_targets"][0, 0], 2.0)


def test_feature_encoder_ignores_hidden_opponent_card_identity() -> None:
    """Verify that hidden opponent card identities do not affect features.

    Inputs:
    - A deterministic x=24 state generated from `build_state_with_remaining_cards`.
    - The same state with only the identities of hidden opponent cards permuted.

    Output:
    - Asserts that both feature vectors are exactly identical.
    """
    state = build_state_with_remaining_cards(24, seed=1_000_001)
    player_id = int(state.turn)
    encoder = SpadesFeatureEncoder()

    original_features = encoder.encode(state, player_id)
    permuted_state = _permute_hidden_opponent_cards(state, player_id=player_id, seed=42)
    permuted_features = encoder.encode(permuted_state, player_id)

    np.testing.assert_array_equal(original_features, permuted_features)