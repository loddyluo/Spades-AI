from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from residual_bidder.actions import BidAction
from rl.nil_first4_observation import (
    NIL_ENCODER_SCHEMA,
    NilFirstFourFeatureEncoderV1,
    build_nil_first_four_observation,
)
from rl.nil_solver_leaf_env import (
    NIL_LOWER,
    NIL_PARTNER,
    NIL_ROLES,
    NIL_SELF,
    NIL_UPPER,
    role_for_seat,
    run_nil_duplicate_candidate,
    run_production_single_nil_auction,
)
from rl.nil_solver_leaf_ppo import (
    export_nil_role_actors,
    load_nil_role_actor_bundle,
    load_nil_training_checkpoint,
    load_nonnil_finetune_weights,
    save_nil_training_checkpoint,
)
from rl.policy_network import PolicyMLP
from rl.run_nil_solver_leaf_convergence import (
    BundleCheckpoint,
    _validate_evaluation_report,
)
from rl.solver_leaf_ppo import (
    PPOConfig,
    ValueMLP,
    build_optimizer,
    save_training_checkpoint,
)
from rl.solver_leaf_env import OpponentPoolConfig
from trick_taking.card import Suit
from trick_taking.game_state import GameState, Phase
from trick_taking.games.spades import SpadesRules


class _SequenceBidder:
    policy_id = "test-residual"

    def __init__(self, actions: list[BidAction], fallback: str | None = None):
        self.actions = list(actions)
        self.fallback = fallback
        self.calls = 0

    def choose(self, state, legal_bids, **kwargs):
        del state, legal_bids, kwargs
        action = self.actions[self.calls]
        self.calls += 1
        return SimpleNamespace(
            action=action,
            fallback_reason=self.fallback,
            effective_policy_id=self.policy_id,
        )


class _SequenceSolver:
    def __init__(self, values: list[float]):
        self.values = list(values)
        self.calls = 0

    def solve(self, state: GameState) -> float:
        assert state.phase is Phase.PLAYING
        assert state.tricks_played == 4
        assert state.table_cards == []
        assert tuple(len(hand) for hand in state.hands) == (9, 9, 9, 9)
        value = self.values[self.calls]
        self.calls += 1
        return value


def _actor(hidden_dims: list[int] | None = None) -> PolicyMLP:
    return PolicyMLP(input_dim=536, hidden_dims=hidden_dims or [64, 32], output_dim=52)


def _actors(hidden_dims: list[int] | None = None) -> dict[str, PolicyMLP]:
    return {role: _actor(hidden_dims) for role in NIL_ROLES}


def _single_nil_auction(shuffle_seed: int = 636_100) -> GameState:
    bidder = _SequenceBidder(
        [BidAction.NIL, BidAction.BID_3, BidAction.BID_4, BidAction.BID_2]
    )
    state, count = run_production_single_nil_auction(
        shuffle_seed, bidder, deal_id="test-single-nil"
    )
    assert state is not None and count == 1
    return state


def test_role_mapping_distinguishes_all_four_positions() -> None:
    assert role_for_seat(0, 0) == NIL_SELF
    assert role_for_seat(0, 1) == NIL_LOWER
    assert role_for_seat(0, 2) == NIL_PARTNER
    assert role_for_seat(0, 3) == NIL_UPPER
    assert {role_for_seat(2, seat) for seat in range(4)} == set(NIL_ROLES)


def test_nil_observation_keeps_536_layout_and_zeroes_only_nil_bid_block() -> None:
    state = _single_nil_auction()
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)
    state.phase = Phase.PLAYING
    state.trump_suit = Suit.SPADES
    seat = state.turn
    legal = rules.playable(state, state.hands[seat], seat)
    observation = build_nil_first_four_observation(state, seat, legal)
    encoder = NilFirstFourFeatureEncoderV1()
    feature = encoder.encode(observation)

    assert encoder.SCHEMA == NIL_ENCODER_SCHEMA
    assert feature.shape == (536,)
    bid_blocks = feature[encoder.BIDS_START : encoder.HISTORY_START].reshape(4, 13)
    assert [int(block.sum()) for block in bid_blocks].count(0) == 1
    assert [int(block.sum()) for block in bid_blocks].count(1) == 3
    assert np.array_equal(
        feature[encoder.LEGAL_START : encoder.BIDS_START].astype(np.bool_),
        np.isin(np.arange(52), observation.legal_card_ids),
    )


def test_auction_accepts_exactly_one_nil_and_rejects_zero_or_multiple() -> None:
    single, count = run_production_single_nil_auction(
        636_101,
        _SequenceBidder(
            [BidAction.BID_2, BidAction.NIL, BidAction.BID_3, BidAction.BID_4]
        ),
        deal_id="single",
    )
    assert single is not None and count == 1

    zero, count = run_production_single_nil_auction(
        636_102,
        _SequenceBidder([BidAction.BID_2] * 4),
        deal_id="zero",
    )
    assert zero is None and count == 0

    multiple, count = run_production_single_nil_auction(
        636_103,
        _SequenceBidder(
            [BidAction.NIL, BidAction.BID_2, BidAction.NIL, BidAction.BID_2]
        ),
        deal_id="multiple",
    )
    assert multiple is None and count == 2

    with pytest.raises(RuntimeError, match="fallback"):
        run_production_single_nil_auction(
            636_104,
            _SequenceBidder([BidAction.NIL] * 4, fallback="synthetic"),
            deal_id="fallback",
        )


def test_nil_duplicate_gives_each_role_four_decisions_and_two_solver_calls() -> None:
    solver = _SequenceSolver([40.0, -20.0])
    outcome = run_nil_duplicate_candidate(
        7,
        636_105,
        _actors(),
        _SequenceBidder(
            [BidAction.NIL, BidAction.BID_3, BidAction.BID_4, BidAction.BID_2]
        ),
        solver,
        NilFirstFourFeatureEncoderV1(),
        run_seed=93,
        deterministic=False,
        record_transitions=True,
    )
    assert outcome.result is not None
    result = outcome.result
    assert solver.calls == 2
    assert result.solver_calls == 2
    assert result.duplicate_margin_points == 30.0
    assert len(result.transitions) == 16
    assert Counter(item.role for item in result.transitions) == Counter(
        {role: 4 for role in NIL_ROLES}
    )
    assert {item.reward for item in result.room_team0.transitions} == {0.4}
    assert {item.reward for item in result.room_team1.transitions} == {0.2}


def test_four_role_checkpoint_export_and_nonnil_initialization(tmp_path: Path) -> None:
    hidden = [64, 32]
    critic_hidden = [32, 16]
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    source_actor = _actor(hidden)
    source_critic = ValueMLP(536, critic_hidden)
    source_optimizer = build_optimizer(source_actor, source_critic, config)
    nonnil = tmp_path / "nonnil.pt"
    save_training_checkpoint(
        nonnil,
        source_actor,
        source_critic,
        source_optimizer,
        update=20,
        deals_trained=40_960,
        candidate_cursor=50_000,
        best_validation_margin=None,
        actor_hidden_dims=hidden,
        critic_hidden_dims=critic_hidden,
        ppo_config=config,
        run_config={"seed": 1},
    )

    actors = _actors(hidden)
    critics = {role: ValueMLP(536, critic_hidden) for role in NIL_ROLES}
    source = load_nonnil_finetune_weights(nonnil, actors, critics)
    assert source.update == 20
    for role in NIL_ROLES:
        for expected, actual in zip(
            source_actor.parameters(), actors[role].parameters(), strict=True
        ):
            assert torch.equal(expected, actual)

    optimizers = {
        role: build_optimizer(actors[role], critics[role], config) for role in NIL_ROLES
    }
    trainer = tmp_path / "nil-trainer.pt"
    save_nil_training_checkpoint(
        trainer,
        actors,
        critics,
        optimizers,
        update=3,
        deals_trained=12,
        candidate_cursor=80,
        actor_hidden_dims=hidden,
        critic_hidden_dims=critic_hidden,
        ppo_config=config,
        run_config={"seed": 2},
    )
    restored_actors = _actors(hidden)
    restored_critics = {role: ValueMLP(536, critic_hidden) for role in NIL_ROLES}
    restored_optimizers = {
        role: build_optimizer(restored_actors[role], restored_critics[role], config)
        for role in NIL_ROLES
    }
    resume = load_nil_training_checkpoint(
        trainer, restored_actors, restored_critics, restored_optimizers
    )
    assert (resume.update, resume.deals_trained, resume.candidate_cursor) == (3, 12, 80)

    manifest = export_nil_role_actors(
        tmp_path,
        actors,
        suffix="final",
        actor_hidden_dims=hidden,
        training_update=3,
        deals_trained=12,
        residual_checkpoint_sha256="a" * 64,
    )
    assert set(manifest["actors"]) == set(NIL_ROLES)
    for role in NIL_ROLES:
        assert (tmp_path / f"actor_{role}_final.pt").is_file()
        assert (tmp_path / f"actor_{role}_final.pt.json").is_file()
    loaded, loaded_manifest, metadata = load_nil_role_actor_bundle(
        tmp_path / "actors_final.json"
    )
    assert loaded_manifest == manifest
    assert set(loaded) == set(NIL_ROLES)
    assert set(metadata) == set(NIL_ROLES)


def test_frozen_nil_bundle_is_used_for_the_opposing_team(tmp_path: Path) -> None:
    frozen = _actors([64, 32])
    export_nil_role_actors(
        tmp_path,
        frozen,
        suffix="frozen",
        actor_hidden_dims=[64, 32],
        training_update=1,
        deals_trained=1,
        residual_checkpoint_sha256="a" * 64,
    )
    loaded, _, _ = load_nil_role_actor_bundle(tmp_path / "actors_frozen.json")
    solver = _SequenceSolver([10.0, -10.0])
    outcome = run_nil_duplicate_candidate(
        9,
        636_106,
        _actors([64, 32]),
        _SequenceBidder(
            [BidAction.BID_2, BidAction.NIL, BidAction.BID_3, BidAction.BID_4]
        ),
        solver,
        NilFirstFourFeatureEncoderV1(),
        run_seed=94,
        deterministic=True,
        record_transitions=True,
        opponent_pool_config=OpponentPoolConfig(
            rule_weight=0.0,
            champion_weight=1.0,
            champion_checkpoint=str(tmp_path / "actors_frozen.json"),
        ),
        opponent_actor_bundles={"champion": loaded},
    )
    assert outcome.result is not None
    assert outcome.result.opponent_id == "champion"
    assert solver.calls == 2
    assert Counter(item.role for item in outcome.result.transitions) == Counter(
        {role: 4 for role in NIL_ROLES}
    )


def test_resumed_evaluation_rejects_stale_configuration(tmp_path: Path) -> None:
    bundle = tmp_path / "actors_final.json"
    bundle.write_text('{"actors": {}, "training_update": 1}\n', encoding="utf-8")
    trainer = tmp_path / "trainer_final.pt"
    trainer.touch()
    checkpoint = BundleCheckpoint(bundle, trainer, 1, 1)
    report = {
        "schema": "solver-leaf-nil-four-role-evaluation-v1",
        "comparison": "candidate-nil-bundle-vs-RuleBasedFirst4NilPlayer",
        "bundle": str(bundle),
        "opponent_bundle": None,
        "bundle_manifest": {"actors": {}, "training_update": 1},
        "opponent_bundle_manifest": None,
        "duplicate_deals": 2,
        "games": 4,
        "solver_calls": 4,
        "workers": 1,
        "seed": 7,
        "base_shuffle_seed": 8,
        "oversample_factor": 6.5,
        "mean_duplicate_margin_points": 1.0,
        "standard_error_points": 0.5,
        "confidence_interval_95_points": [0.02, 1.98],
        "wins": 1,
        "ties": 1,
        "losses": 0,
    }
    assert _validate_evaluation_report(
        report,
        checkpoint,
        None,
        deals=2,
        workers=1,
        seed=7,
        base_seed=8,
        oversample_factor=6.5,
    ) is report
    stale = dict(report, seed=9)
    with pytest.raises(ValueError, match="seed mismatch"):
        _validate_evaluation_report(
            stale,
            checkpoint,
            None,
            deals=2,
            workers=1,
            seed=7,
            base_seed=8,
            oversample_factor=6.5,
        )
