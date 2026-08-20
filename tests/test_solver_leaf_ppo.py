from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from residual_bidder.actions import BidAction
from rl.first4_observation import FirstFourFeatureEncoderV2
from rl.policy_network import PolicyMLP
from rl.solver_leaf_env import (
    OpponentPoolConfig,
    derive_action_seed,
    legal_mask_from_ids,
    masked_action_probabilities,
    run_duplicate_candidate,
    run_production_auction,
    select_opponent_id,
    select_policy_action,
)
from rl.solver_leaf_ppo import (
    PPOConfig,
    ValueMLP,
    build_optimizer,
    export_actor,
    load_exported_actor,
    load_finetune_weights,
    load_training_checkpoint,
    policy_log_probs_and_entropy,
    ppo_update,
    save_training_checkpoint,
    stack_transitions,
)
from trick_taking.game_state import GameState, Phase


class _FixedBidder:
    policy_id = "test-residual"

    def __init__(self, action: BidAction = BidAction.BID_3, fallback: str | None = None):
        self.action = action
        self.fallback = fallback

    def choose(self, state, legal_bids, **kwargs):
        del state, legal_bids, kwargs
        return SimpleNamespace(
            action=self.action,
            fallback_reason=self.fallback,
            effective_policy_id=self.policy_id,
        )


class _SequenceSolver:
    def __init__(self, values: list[float]):
        self.values = list(values)
        self.calls = 0
        self.boundaries: list[tuple[int, tuple[int, ...]]] = []

    def solve(self, state: GameState) -> float:
        assert state.phase is Phase.PLAYING
        assert state.tricks_played == 4
        assert state.table_cards == []
        assert tuple(len(hand) for hand in state.hands) == (9, 9, 9, 9)
        self.boundaries.append((state.tricks_played, tuple(map(len, state.hands))))
        value = self.values[self.calls]
        self.calls += 1
        return value


def _actor(hidden_dims: list[int] | None = None) -> PolicyMLP:
    return PolicyMLP(input_dim=536, hidden_dims=hidden_dims or [64, 32], output_dim=52)


def _duplicate(actor: PolicyMLP | None = None):
    solver = _SequenceSolver([40.0, -20.0])
    outcome = run_duplicate_candidate(
        7,
        536100,
        actor or _actor(),
        _FixedBidder(),
        solver,
        FirstFourFeatureEncoderV2(),
        run_seed=91,
        deterministic=False,
        record_transitions=True,
    )
    assert outcome.result is not None
    return outcome.result, solver


def test_duplicate_calls_solver_once_per_room_and_applies_team_sign_and_divisor() -> None:
    result, solver = _duplicate()

    assert solver.calls == 2
    assert result.solver_calls == 2
    assert result.room_team0.team0_margin_points == 40.0
    assert result.room_team0.candidate_margin_points == 40.0
    assert result.room_team0.reward == 0.4
    assert result.room_team1.team0_margin_points == -20.0
    assert result.room_team1.candidate_margin_points == 20.0
    assert result.room_team1.reward == 0.2
    assert result.duplicate_margin_points == 30.0
    assert len(result.room_team0.transitions) == 8
    assert len(result.room_team1.transitions) == 8
    assert {item.reward for item in result.room_team0.transitions} == {0.4}
    assert {item.reward for item in result.room_team1.transitions} == {0.2}


def test_nil_auction_is_filtered_before_solver_and_fallback_is_fatal() -> None:
    actor = _actor()
    solver = _SequenceSolver([])
    outcome = run_duplicate_candidate(
        8,
        536101,
        actor,
        _FixedBidder(BidAction.NIL),
        solver,
        FirstFourFeatureEncoderV2(),
        run_seed=92,
        deterministic=False,
        record_transitions=True,
    )
    assert outcome.nil_filtered is True
    assert outcome.result is None
    assert solver.calls == 0

    with pytest.raises(RuntimeError, match="fallback"):
        run_production_auction(
            536102,
            _FixedBidder(fallback="synthetic-fallback"),
            deal_id="fallback-test",
        )


def test_masked_policy_assigns_zero_probability_and_argmax_uses_lowest_card_id() -> None:
    actor = _actor([16])
    for parameter in actor.parameters():
        torch.nn.init.zeros_(parameter)
    feature = np.zeros(536, dtype=np.float32)
    legal_mask = legal_mask_from_ids([7, 19, 51])
    feature[52:104] = legal_mask.astype(np.float32)
    probabilities = masked_action_probabilities(
        actor,
        torch.from_numpy(feature),
        torch.from_numpy(legal_mask),
    )

    assert float(probabilities[~torch.from_numpy(legal_mask)].sum().item()) == 0.0
    assert float(probabilities[torch.from_numpy(legal_mask)].sum().item()) == pytest.approx(1.0)
    action, _, _ = select_policy_action(
        actor, feature, legal_mask, deterministic=True, sample_seed=1
    )
    assert action == 7


def test_action_seed_and_duplicate_sampling_are_reproducible() -> None:
    assert derive_action_seed(1, 2, 0, 3, 4) == derive_action_seed(1, 2, 0, 3, 4)
    assert derive_action_seed(1, 2, 0, 3, 4) != derive_action_seed(1, 2, 1, 3, 4)
    actor = _actor()
    first, _ = _duplicate(actor)
    second, _ = _duplicate(actor)
    assert [item.action for item in first.transitions] == [
        item.action for item in second.transitions
    ]
    for left, right in zip(first.transitions, second.transitions, strict=True):
        assert np.array_equal(left.feature, right.feature)
        assert np.array_equal(left.legal_mask, right.legal_mask)


def test_opponent_pool_selection_is_deterministic_and_tracks_requested_mix() -> None:
    config = OpponentPoolConfig(
        rule_weight=0.60,
        champion_weight=0.25,
        history_weight=0.15,
        champion_checkpoint="champion.pt",
        history_checkpoints=("history-a.pt", "history-b.pt"),
    )
    first = [
        select_opponent_id(config, run_seed=75, candidate_index=index)
        for index in range(10_000)
    ]
    second = [
        select_opponent_id(config, run_seed=75, candidate_index=index)
        for index in range(10_000)
    ]
    assert first == second
    counts = Counter(first)
    assert counts["rule"] / len(first) == pytest.approx(0.60, abs=0.02)
    assert counts["champion"] / len(first) == pytest.approx(0.25, abs=0.02)
    history_fraction = (counts["history:0"] + counts["history:1"]) / len(first)
    assert history_fraction == pytest.approx(0.15, abs=0.02)


def test_frozen_policy_opponent_is_used_in_both_duplicate_rooms() -> None:
    learner = _actor()
    frozen = _actor()
    for parameter in frozen.parameters():
        torch.nn.init.zeros_(parameter)
    solver = _SequenceSolver([10.0, -10.0])
    outcome = run_duplicate_candidate(
        7,
        536100,
        learner,
        _FixedBidder(),
        solver,
        FirstFourFeatureEncoderV2(),
        run_seed=91,
        deterministic=False,
        record_transitions=True,
        opponent_pool_config=OpponentPoolConfig(
            rule_weight=0.0,
            champion_weight=1.0,
            champion_checkpoint="unused-in-direct-test.pt",
        ),
        opponent_actors={"champion": frozen},
    )
    assert outcome.result is not None
    assert outcome.result.opponent_id == "champion"
    assert solver.calls == 2
    assert len(outcome.result.transitions) == 16


def test_ppo_ratio_starts_at_one_and_update_is_finite() -> None:
    actor = _actor()
    critic = ValueMLP(536, [32, 16])
    result, _ = _duplicate(actor)
    batch = stack_transitions(result.transitions, device="cpu")
    with torch.no_grad():
        current, _ = policy_log_probs_and_entropy(
            actor, batch.features, batch.legal_masks, batch.actions
        )
    assert torch.allclose(current, batch.old_log_probs, atol=1e-6, rtol=0.0)

    config = PPOConfig(update_epochs=2, minibatch_size=8)
    optimizer = build_optimizer(actor, critic, config)
    stats = ppo_update(
        actor,
        critic,
        optimizer,
        result.transitions,
        config,
        device="cpu",
        shuffle_seed=3,
    )
    numeric = [
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.approximate_kl,
        stats.clip_fraction,
        stats.gradient_norm,
    ]
    assert all(np.isfinite(value) for value in numeric)
    assert stats.transitions == 16


def test_ppo_reports_clipping_for_intentionally_stale_log_probabilities() -> None:
    actor = _actor()
    critic = ValueMLP(536, [32, 16])
    result, _ = _duplicate(actor)
    stale = tuple(
        replace(item, old_log_prob=item.old_log_prob + 2.0)
        for item in result.transitions
    )
    config = PPOConfig(update_epochs=1, minibatch_size=16)
    stats = ppo_update(
        actor,
        critic,
        build_optimizer(actor, critic, config),
        stale,
        config,
        device="cpu",
        shuffle_seed=4,
    )
    assert stats.clip_fraction == 1.0


def test_training_checkpoint_and_actor_export_round_trip(tmp_path: Path) -> None:
    actor = _actor()
    critic = ValueMLP(536, [32, 16])
    config = PPOConfig(update_epochs=1, minibatch_size=16)
    optimizer = build_optimizer(actor, critic, config)
    trainer_path = tmp_path / "trainer.pt"
    save_training_checkpoint(
        trainer_path,
        actor,
        critic,
        optimizer,
        update=5,
        deals_trained=64,
        candidate_cursor=71,
        best_validation_margin=None,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[32, 16],
        ppo_config=config,
        run_config={"seed": 9},
    )

    restored_actor = _actor()
    restored_critic = ValueMLP(536, [32, 16])
    restored_optimizer = build_optimizer(restored_actor, restored_critic, config)
    resume = load_training_checkpoint(
        trainer_path, restored_actor, restored_critic, restored_optimizer
    )
    assert (resume.update, resume.deals_trained, resume.candidate_cursor) == (5, 64, 71)
    for expected, actual in zip(actor.parameters(), restored_actor.parameters(), strict=True):
        assert torch.equal(expected, actual)

    finetune_actor = _actor()
    finetune_critic = ValueMLP(536, [32, 16])
    source = load_finetune_weights(trainer_path, finetune_actor, finetune_critic)
    assert (source.update, source.deals_trained, source.candidate_cursor) == (5, 64, 71)
    fresh_optimizer = build_optimizer(finetune_actor, finetune_critic, config)
    assert fresh_optimizer.state == {}
    for expected, actual in zip(actor.parameters(), finetune_actor.parameters(), strict=True):
        assert torch.equal(expected, actual)

    actor_path = tmp_path / "actor.pt"
    metadata = export_actor(
        actor_path,
        actor,
        actor_hidden_dims=[64, 32],
        training_update=5,
        deals_trained=64,
        residual_checkpoint_sha256="a" * 64,
    )
    loaded_actor, loaded_metadata = load_exported_actor(actor_path)
    assert loaded_metadata == metadata
    probe = torch.zeros(536)
    with torch.no_grad():
        assert torch.equal(actor(probe), loaded_actor(probe))
