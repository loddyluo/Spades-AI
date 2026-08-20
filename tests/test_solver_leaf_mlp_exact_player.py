from __future__ import annotations

import pytest
import torch

from rl.nil_first4_observation import (
    NilFirstFourFeatureEncoderV1,
    build_nil_first_four_observation,
)
from rl.nil_solver_leaf_env import NIL_ROLES
from rl.policy_network import PolicyMLP
from strategy.solver_leaf_mlp_exact_player import (
    SolverLeafMLPExactPlayer,
    role_for_nil_configuration,
)
from strategy.spades_match_runner import build_random_state
from trick_taking.card import Suit
from trick_taking.game_state import Bid, Phase
from trick_taking.games.spades import SpadesRules


HASH = "a" * 64


def _actor(preferred_card_id: int) -> PolicyMLP:
    actor = PolicyMLP(input_dim=536, hidden_dims=[4], output_dim=52)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.policy_head.bias[preferred_card_id] = 20.0
    return actor.eval()


def _playing_state(*, seat: int, bids: list[str]):
    state = build_random_state(2026082001 + seat)
    state.phase = Phase.PLAYING
    state.teams = [0, 1, 0, 1]
    state.max_bid = list(bids)
    state.bids = [
        Bid(player_id=bidder, value=value, is_pass=False)
        for bidder, value in enumerate(bids)
    ]
    state.trump_suit = Suit.SPADES
    state.turn = state.trick_leader = seat
    legal = sorted(
        SpadesRules().playable(state, state.hands[seat], seat),
        key=lambda card: card.card_id,
    )
    assert len(legal) >= 5
    return state, legal


def _player_for_legal(legal):
    nonnil = _actor(legal[0].card_id)
    nil_actors = {
        role: _actor(legal[index + 1].card_id)
        for index, role in enumerate(NIL_ROLES)
    }
    return SolverLeafMLPExactPlayer(
        nonnil_actor=nonnil,
        nonnil_model_id="nonnil-test",
        nonnil_actor_sha256=HASH,
        nil_actors=nil_actors,
        nil_model_id="nil-test",
        nil_bundle_sha256=HASH,
        exact_solver=object(),
        num_workers=1,
    )


@pytest.mark.parametrize(
    ("nil_seats", "expected"),
    [
        ((0,), ("nil_self", "nil_lower", "nil_partner", "nil_upper")),
        ((0, 1), ("nil_self", "nil_self", "nil_partner", "nil_partner")),
        ((0, 2), ("nil_self", "nil_lower", "nil_self", "nil_upper")),
        ((1, 3), ("nil_upper", "nil_self", "nil_lower", "nil_self")),
        ((0, 1, 2), ("nil_self", "nil_self", "nil_self", "nil_partner")),
        ((0, 1, 2, 3), ("nil_self",) * 4),
    ],
)
def test_role_mapping_covers_single_and_multi_nil(nil_seats, expected) -> None:
    assert tuple(
        role_for_nil_configuration(nil_seats, seat) for seat in range(4)
    ) == expected


def test_no_nil_first_four_uses_nonnil_actor() -> None:
    state, legal = _playing_state(seat=0, bids=["bid_3"] * 4)
    player = _player_for_legal(legal)
    player.start_game(0, list(state.hands[0]), 4)
    player.set_teams(state.teams, state.max_bid)

    chosen = player.play_card(legal, {"state": state})

    assert chosen == legal[0]
    assert player.last_play_info["mode"] == "solver_leaf_mlp_first4"
    assert player.last_play_info["role"] == "nonnil"


@pytest.mark.parametrize(
    ("seat", "bids", "expected_role"),
    [
        (0, ["nil", "bid_3", "bid_3", "bid_3"], "nil_self"),
        (0, ["blind_nil", "bid_3", "bid_3", "bid_3"], "nil_self"),
        (2, ["nil", "bid_3", "bid_3", "bid_3"], "nil_partner"),
        (2, ["nil", "nil", "bid_3", "bid_3"], "nil_partner"),
        (1, ["nil", "bid_3", "nil", "bid_3"], "nil_lower"),
        (3, ["nil", "bid_3", "nil", "bid_3"], "nil_upper"),
        (3, ["nil", "nil", "nil", "bid_3"], "nil_partner"),
    ],
)
def test_nil_first_four_uses_requested_role_actor(
    seat: int,
    bids: list[str],
    expected_role: str,
) -> None:
    state, legal = _playing_state(seat=seat, bids=bids)
    player = _player_for_legal(legal)
    player.start_game(seat, list(state.hands[seat]), 4)
    player.set_teams(state.teams, state.max_bid)

    chosen = player.play_card(legal, {"state": state})

    expected_index = list(NIL_ROLES).index(expected_role) + 1
    assert chosen == legal[expected_index]
    assert player.last_play_info["mode"] == "nil_solver_leaf_mlp_first4"
    assert player.last_play_info["role"] == expected_role


def test_multi_nil_observation_uses_one_zero_bid_block_per_nil() -> None:
    bids = ["nil", "bid_3", "nil", "bid_4"]
    state, legal = _playing_state(seat=1, bids=bids)
    observation = build_nil_first_four_observation(state, 1, legal)
    feature = NilFirstFourFeatureEncoderV1().encode(observation)
    bid_blocks = feature[104:156].reshape(4, 13)

    assert [int(block.sum()) for block in bid_blocks].count(0) == 2


def test_posterior_replay_uses_mlp_for_no_nil() -> None:
    state, legal = _playing_state(seat=2, bids=["bid_2"] * 4)
    player = _player_for_legal(legal)
    context = player._create_first4_replay_player(
        2,
        list(state.hands[2]),
        state.max_bid,
    )

    chosen = player._first4_replay_expected_card(
        context,
        legal,
        {},
        player_id=2,
        current_hand=list(state.hands[2]),
        prior_plays=[],
        max_bid=state.max_bid,
    )

    assert chosen == legal[0]


def test_player_rejects_a_boundary_that_would_reintroduce_fallback() -> None:
    actors = {role: _actor(0) for role in NIL_ROLES}
    with pytest.raises(ValueError, match="at least 36"):
        SolverLeafMLPExactPlayer(
            nonnil_actor=_actor(0),
            nonnil_model_id="nonnil-test",
            nonnil_actor_sha256=HASH,
            nil_actors=actors,
            nil_model_id="nil-test",
            nil_bundle_sha256=HASH,
            exact_solver=object(),
            exact_threshold=35,
        )
