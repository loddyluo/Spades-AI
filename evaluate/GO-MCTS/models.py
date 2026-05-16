"""Model exports and checkpoint loaders for the collaborator GO-MCTS repo.

File purpose:
- Re-export the collaborator repository's player/model classes behind a local
  evaluation import path.
- Provide lightweight checkpoint loaders for the GPT-2 policy/value model and
  the bid MLP model.

Function input/output summary:
- load_gpt2_policy_value_model(checkpoint_path: str, device: str) -> GPT2PolicyValueModel
    Input: path to a compatible `.pt` checkpoint and the target device name.
    Output: a loaded GPT2PolicyValueModel in eval mode.
- load_bid_mlp_model(checkpoint_path: str, device: str) -> BidMLP
    Input: path to a compatible `.pt` checkpoint and the target device name.
    Output: a loaded BidMLP in eval mode.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COLLAB_ROOT = _REPO_ROOT / "Spades_AI_GO-MCTS"
if str(_COLLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_COLLAB_ROOT))

from spades_ai.models.bid_mlp import BidMLP
from spades_ai.players.argmax_player import ArgmaxPlayer
from spades_ai.players.gomcts_player import GOMCTSPlayer
from spades_ai.players.mlp_bid_player import MLPBidPlayer
from spades_ai.players.random_player import RandomPlayer
from spades_ai.players.rule_based.player import RuleBasedPlayer
from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as RuleBasedPlayerV2
from spades_ai.search.go_mcts import GOMCTSConfig


def load_gpt2_policy_value_model(checkpoint_path: str, device: str):
    """Load the collaborator GPT-2 policy/value checkpoint.

    Input:
    - checkpoint_path: file path to a checkpoint produced by the collaborator
      repository's training code.
    - device: target torch device string, for example "cpu" or "cuda:0".

    Output:
    - A GPT2PolicyValueModel moved to `device`, set to eval mode, and frozen.
    """
    from spades_ai.models.gpt2_policy_value import GPT2PolicyValueConfig, GPT2PolicyValueModel

    config = GPT2PolicyValueConfig(
        vocab_size=438,
        n_positions=512,
        n_embd=256,
        n_head=8,
        n_layer=8,
        num_labels=721,
    )
    model = GPT2PolicyValueModel(config)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.requires_grad_(False)
    model.train(False)
    return model


def load_bid_mlp_model(checkpoint_path: str, device: str) -> BidMLP:
    """Load the collaborator bid MLP checkpoint.

    Input:
    - checkpoint_path: file path to a BidMLP checkpoint.
    - device: target torch device string, for example "cpu" or "cuda:0".

    Output:
    - A BidMLP moved to `device`, set to eval mode, and frozen.
    """
    model = BidMLP()
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.requires_grad_(False)
    model.train(False)
    return model


__all__ = [
    "ArgmaxPlayer",
    "BidMLP",
    "GOMCTSConfig",
    "GOMCTSPlayer",
    "MLPBidPlayer",
    "RandomPlayer",
    "RuleBasedPlayer",
    "RuleBasedPlayerV2",
    "load_bid_mlp_model",
    "load_gpt2_policy_value_model",
]
