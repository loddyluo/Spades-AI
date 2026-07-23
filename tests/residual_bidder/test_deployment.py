from __future__ import annotations

import pytest

import residual_bidder.deployment as deployment_module
from residual_bidder.deployment import (
    DEPLOYED_CALIBRATION,
    DEPLOYED_CHECKPOINT_SHA256,
    DEPLOYED_MODEL_ID,
    load_deployed_acting_bidder,
)
from residual_bidder.hybrid import _initial_state
from residual_bidder.random_tape import BidSamplingKey, policy_uniform
from trick_taking.game_state import Bid
from trick_taking.games.spades import SpadesRules


def test_selected_checkpoint_loads_as_deterministic_acting_bidder() -> None:
    bidder = load_deployed_acting_bidder()
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state = _initial_state(202607220001, rules)
    seat = state.current_bidder
    legal_bids = rules.legal_bids(state, seat)

    first = bidder.choose(
        state,
        legal_bids,
        logical_seat=seat,
        deal_id="deployment-test",
        room_id="room-a",
    )
    second = bidder.choose(
        state,
        legal_bids,
        logical_seat=seat,
        deal_id="deployment-test",
        room_id="room-a",
    )

    assert bidder.model_id == DEPLOYED_MODEL_ID
    assert bidder.checkpoint_sha256 == DEPLOYED_CHECKPOINT_SHA256
    assert bidder.policy.calibration == DEPLOYED_CALIBRATION
    assert bidder.policy_id == (
        "18963ba56a997f2e70ba05f8f649072390e07d1fb5d0a181a678bdda0b9e1a52"
    )
    assert bidder.describe()["calibration"] == {
        "uncertainty_lambda": 1.0,
        "temperature": 0.0,
        "epsilon": 0.0,
        "rho": 1.0,
    }
    assert bidder.describe()["belief_bidder"] == "bid_nsfp.pt"
    assert bidder.describe()["training_play_pipeline_sha256"] == (
        "8bfe50f89f77f50eb6489576d41949bd31565ccb1d3f2b246e4dcba7718d2c39"
    )
    assert first.action is second.action
    assert first.fallback_reason is None
    assert sum(value > 0.0 for value in first.distribution.probabilities) == 1
    assert abs(int(first.action) - int(first.distribution.center)) <= 1


def test_selected_checkpoint_is_hash_pinned_before_deserialization() -> None:
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_deployed_acting_bidder(expected_checkpoint_sha256="0" * 64)


def test_deployment_does_not_inspect_runtime_card_play_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_runtime_hash(*_args, **_kwargs):
        raise AssertionError("production load must not hash card-play sources")

    monkeypatch.setattr(
        deployment_module,
        "_play_pipeline_sha256",
        unexpected_runtime_hash,
    )

    bidder = load_deployed_acting_bidder()

    assert bidder.model_id == DEPLOYED_MODEL_ID


def test_declining_blind_nil_does_not_change_actual_bid_index() -> None:
    bidder = load_deployed_acting_bidder()
    rules = SpadesRules(enable_nil=True, enable_blind_nil=True)
    state = _initial_state(202607220004, rules)
    seat = state.current_bidder
    state.bids.append(Bid(player_id=seat, value="pass", is_pass=True))

    decision = bidder.choose(
        state,
        rules.legal_bids(state, seat),
        logical_seat=seat,
        deal_id="blind-pass-test",
        room_id="room-b",
    )

    assert decision.uniform == policy_uniform(BidSamplingKey(
        policy_seed=bidder.policy_seed,
        deal_id="blind-pass-test",
        room_id="room-b",
        logical_seat=seat,
        bid_index=0,
    ))
