from __future__ import annotations

import copy

import numpy as np
import pytest

from rl.first4_observation import (
    FirstFourFeatureEncoderV2,
    build_first_four_observation,
)
from strategy.spades_match_runner import build_random_state
from trick_taking.card import Suit, cards_to_bitset
from trick_taking.game_state import Bid, Phase
from trick_taking.games.spades import SpadesRules


def _playing_state(seed: int = 536001):
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state = build_random_state(seed)
    for bid_number in (3, 4, 2, 5):
        seat = state.current_bidder
        value = f"bid_{bid_number}"
        state.bids.append(Bid(player_id=seat, value=value, is_pass=False))
        state.max_bid[seat] = value
        state.current_bidder = rules.next_bid_turn(state)
    state.teams = rules.set_team(state)
    state.points = rules.initial_points(state)
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    return state, rules


def test_encoder_has_exact_536_layout_and_one_hot_sections() -> None:
    state, rules = _playing_state()
    seat = state.turn
    legal = rules.playable(state, state.hands[seat], seat)
    observation = build_first_four_observation(state, seat, legal)
    encoder = FirstFourFeatureEncoderV2()

    feature = encoder.encode(observation)

    assert feature.shape == (536,)
    assert feature.dtype == np.float32
    assert np.isfinite(feature).all()
    assert np.isin(feature[0:509], (0.0, 1.0)).all()
    assert np.isin(feature[511:524], (0.0, 1.0)).all()
    assert feature[0:52].sum() == 13
    assert feature[52:104].sum() == len(legal)
    assert feature[104:156].sum() == 4
    assert feature[156:471].sum() == 0
    assert feature[471:493].sum() == 5
    assert feature[493:509].sum() == 4
    assert np.count_nonzero(feature[511:523]) == 0
    assert feature[523] == 0
    assert np.logical_and(feature[509:511] >= -1.0, feature[509:511] <= 1.0).all()
    assert np.logical_and(feature[524:536] >= 0.0, feature[524:536] <= 1.0).all()
    assert feature[524:528].sum() == pytest.approx(1.0)
    assert np.isin(feature[532:536], (0.0, 1.0)).all()
    assert FirstFourFeatureEncoderV2.segment_ranges()["hand_derived"] == (524, 536)


def test_history_slot_is_factorized_rank_suit_and_relative_seat_one_hot() -> None:
    state, rules = _playing_state(536002)
    first = state.turn
    legal = rules.playable(state, state.hands[first], first)
    card = min(legal, key=lambda item: item.card_id)
    state.play_card_to_table(first, card)
    if card.suit == Suit.SPADES:
        state.spades_broken = state.trump_broken = True
    state.turn = (first + 1) % 4
    second = state.turn
    second_legal = rules.playable(state, state.hands[second], second)

    observation = build_first_four_observation(state, second, second_legal)
    feature = FirstFourFeatureEncoderV2().encode(observation)
    slot = feature[156 : 156 + 21]

    assert slot[0:13].sum() == 1
    assert slot[13:17].sum() == 1
    assert slot[17:21].sum() == 1
    assert slot[card.rank.value - 2] == 1
    assert slot[13 + card.suit.value] == 1
    assert feature[156 + 21 : 471].sum() == 0


def test_permuting_all_concealed_opponent_hands_does_not_change_observation() -> None:
    state, rules = _playing_state(536003)
    seat = state.turn
    legal = rules.playable(state, state.hands[seat], seat)
    original_observation = build_first_four_observation(state, seat, legal)
    original_feature = FirstFourFeatureEncoderV2().encode(original_observation)

    permuted = copy.deepcopy(state)
    opponents = [value for value in range(4) if value != seat]
    rotated = [list(permuted.hands[value]) for value in opponents[1:] + opponents[:1]]
    for target, hand in zip(opponents, rotated, strict=True):
        permuted.hands[target] = hand
        permuted.hand_bitsets[target] = cards_to_bitset(hand)
    permuted_legal = rules.playable(permuted, permuted.hands[seat], seat)
    permuted_observation = build_first_four_observation(permuted, seat, permuted_legal)
    permuted_feature = FirstFourFeatureEncoderV2().encode(permuted_observation)

    assert original_observation == permuted_observation
    assert np.array_equal(original_feature, permuted_feature)


def test_public_void_feature_uses_only_failure_to_follow_suit() -> None:
    state, rules = _playing_state(536010)
    leader = state.turn
    diamond = next(card for card in state.hands[leader] if card.suit == Suit.DIAMONDS)
    assert diamond in rules.playable(state, state.hands[leader], leader)
    state.play_card_to_table(leader, diamond)
    state.turn = (leader + 1) % 4
    void_seat = state.turn
    assert not any(card.suit == Suit.DIAMONDS for card in state.hands[void_seat])
    discard = rules.playable(state, state.hands[void_seat], void_seat)[0]
    state.play_card_to_table(void_seat, discard)
    if discard.suit == Suit.SPADES:
        state.spades_broken = state.trump_broken = True
    state.turn = (void_seat + 1) % 4
    observer = state.turn
    legal = rules.playable(state, state.hands[observer], observer)

    observation = build_first_four_observation(state, observer, legal)
    feature = FirstFourFeatureEncoderV2().encode(observation)

    # From observer seat 3, void_seat 2 is relative seat 1, the first void row.
    assert observation.public_voids[0][Suit.DIAMONDS.value] is True
    assert feature[FirstFourFeatureEncoderV2.VOIDS_START + Suit.DIAMONDS.value] == 1


def test_nil_bid_is_rejected_by_non_nil_encoder() -> None:
    state, rules = _playing_state(536004)
    state.max_bid[0] = "nil"
    seat = state.turn
    legal = rules.playable(state, state.hands[seat], seat)

    with pytest.raises(ValueError, match="non-Nil numeric bid"):
        build_first_four_observation(state, seat, legal)
