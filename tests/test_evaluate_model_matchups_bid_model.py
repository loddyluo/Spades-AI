"""Tests for bid-MLP integration in evaluate_model_matchups.

These tests check that `--bid-checkpoint` is loaded into the runtime as
`bid_model` and that `build_players` uses the MLP bidding wrapper for
`our_mcts` when the model is present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import build_runtime, build_players


def build_args_with_bid() -> argparse.Namespace:
    return argparse.Namespace(
        device="cpu",
        p0="our_mcts",
        p1="go_rule",
        p2="go_rule",
        p3="go_random",
        our_checkpoint="",
        our_prior_oracle_spec="no",
        our_exact_threshold=24,
        our_leaf_threshold=24,
        our_simulations_per_action=11,
        our_mcts_determinization_count=5,
        our_number_of_exact_solvers=50,
        our_exploration_constant=1.5,
        our_policy_temperature=1.0,
        our_value_scale=25.0,
        go_pv_checkpoint="",
        go_bid_checkpoint="",
        go_mcts_runs=100,
        go_mcts_steps=5,
        go_mcts_c=0.3,
        go_mcts_mu=0.01,
        go_mcts_threshold=0.05,
        go_argmax_threshold=0.05,
        bid_checkpoint="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt",
    )


def test_build_runtime_loads_bid_model() -> None:
    args = build_args_with_bid()
    runtime = build_runtime(args)
    assert getattr(runtime, "bid_model", None) is not None


def test_build_players_uses_mlp_bid_wrapper(tmp_path) -> None:
    args = build_args_with_bid()
    runtime = build_runtime(args)
    players = build_players(args, runtime, game_seed=0)
    # seat 0 should be our_mcts and use the MLP bidder wrapper when bid_model present
    our0 = players[0]
    # The wrapper class is defined in the evaluate module; ensure place_bid exists
    assert hasattr(our0, "place_bid")
    # ensuring it's not the pure OurHandStrengthMCTSPlayer by checking for _mlp_bidder
    assert hasattr(our0, "_mlp_bidder")
