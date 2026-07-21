from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest
import torch
from torch import nn

from residual_bidder.actions import BidAction
from residual_bidder.nsfp import FrozenNSFP
from spades_ai.models.bid_encoder import BidEncoder
from trick_taking.card import Card, Rank, Suit
from trick_taking.game_state import GameState, Phase


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt"
CHECKPOINT_SHA256 = "b994add8b3a7067aac95000f8f61f90df9045f5a3d18be5fbc947756a7c2066c"


def _load_bridge():
    path = REPO_ROOT / "evaluate/GO-MCTS/bridge.py"
    spec = importlib.util.spec_from_file_location("task2_reference_bridge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE_BRIDGE = _load_bridge()


def _state(dealer: int, bidding_position: int) -> GameState:
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    random.Random(10_000 + 10 * dealer + bidding_position).shuffle(deck)
    hands = [deck[index * 13 : (index + 1) * 13] for index in range(4)]
    prior = ["nil", "blind_nil", "bid_4"]
    max_bid: list[str | None] = prior[:bidding_position] + [None] * (4 - bidding_position)
    return GameState(
        num_players=4,
        phase=Phase.BIDDING,
        dealer_seat=dealer,
        hands=hands,
        hand_bitsets=[0, 0, 0, 0],
        all_cards=deck,
        max_bid=max_bid,
        current_bidder=bidding_position,
        teams=[0, 1, 0, 1],
        tricks_won=[0, 0, 0, 0],
    )


@pytest.fixture(scope="module")
def frozen() -> FrozenNSFP:
    return FrozenNSFP.load(CHECKPOINT, CHECKPOINT_SHA256, torch.device("cpu"))


@pytest.mark.parametrize("dealer", range(4))
@pytest.mark.parametrize("bidding_position", range(4))
def test_observation_is_bit_exact_to_frozen_bridge_encoder_path(
    frozen: FrozenNSFP, dealer: int, bidding_position: int
) -> None:
    state = _state(dealer, bidding_position)
    go_state = REFERENCE_BRIDGE.to_go_state(state)
    expected = BidEncoder().encode(
        list(go_state.hands[go_state.current_player]),
        list(go_state.bids),
        len(go_state.bids),
    )

    observation = frozen.observe(state)

    assert observation.encoded_149.shape == (149,)
    assert torch.equal(observation.encoded_149, expected)
    assert observation.raw_logits_16.shape == (16,)
    assert observation.legal_scores_14.shape == (14,)
    assert isinstance(observation.center, BidAction)


def test_nonacting_private_hands_do_not_affect_observation_or_outputs(
    frozen: FrozenNSFP,
) -> None:
    state = _state(dealer=2, bidding_position=1)
    mutated = _state(dealer=2, bidding_position=1)
    mutated.hands = [list(hand) for hand in state.hands]
    nonactors = [seat for seat in range(4) if seat != state.current_bidder]
    rotated = [state.hands[seat] for seat in nonactors[1:] + nonactors[:1]]
    for seat, replacement in zip(nonactors, rotated, strict=True):
        mutated.hands[seat] = list(replacement)

    original = frozen.observe(state)
    changed = frozen.observe(mutated)

    assert torch.equal(changed.encoded_149, original.encoded_149)
    assert torch.equal(changed.raw_logits_16, original.raw_logits_16)
    assert torch.equal(changed.legal_scores_14, original.legal_scores_14)
    assert changed.center is original.center


def test_observe_batch_matches_individual_observations(frozen: FrozenNSFP) -> None:
    states = [_state(dealer, position) for dealer in range(4) for position in range(4)]

    batched = frozen.observe_batch(states)
    individual = [frozen.observe(state) for state in states]

    assert len(batched) == len(states)
    for actual, expected in zip(batched, individual, strict=True):
        assert torch.equal(actual.encoded_149, expected.encoded_149)
        torch.testing.assert_close(actual.raw_logits_16, expected.raw_logits_16)
        torch.testing.assert_close(actual.legal_scores_14, expected.legal_scores_14)
        assert actual.center is expected.center
    assert frozen.observe_batch([]) == []


def test_load_verifies_hash_before_deserialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drifted = tmp_path / "bid_nsfp.pt"
    drifted.write_bytes(b"drifted checkpoint")
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before hash verification")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenNSFP.load(drifted, CHECKPOINT_SHA256, torch.device("cpu"))
    assert called is False


def test_loaded_model_is_frozen_and_in_evaluation_mode(frozen: FrozenNSFP) -> None:
    assert frozen.model.training is False
    assert all(parameter.requires_grad is False for parameter in frozen.model.parameters())


class _BadOutputModel(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output.to(inputs.device)


@pytest.mark.parametrize(
    "bad_output",
    [torch.zeros(16), torch.zeros(1, 15), torch.full((1, 16), float("nan"))],
)
def test_observe_rejects_invalid_model_outputs(
    frozen: FrozenNSFP, bad_output: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(frozen, "model", _BadOutputModel(bad_output))

    with pytest.raises(ValueError):
        frozen.observe(_state(dealer=0, bidding_position=0))


def test_observation_does_not_expose_hidden_activations(frozen: FrozenNSFP) -> None:
    observation = frozen.observe(_state(dealer=0, bidding_position=0))

    assert set(vars(observation)) == {
        "encoded_149",
        "raw_logits_16",
        "legal_scores_14",
        "center",
    }
