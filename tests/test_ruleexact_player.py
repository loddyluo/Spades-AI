"""Tests for the `ruleexact` hybrid player.

File purpose:
- Verify the early-game behavior matches `rule_based_v2` exactly.
- Verify the `TruncatedMCTSStrategy` inside `RuleExactPlayer` is configured
  with the `go_rule_2` prior so the importance-sampling code does not fall
  back to the uniform-only path.

Test I/O summary:
- `test_ruleexact_matches_rule_based_v2_in_early_play`
    Input: a seeded local Spades deal after bidding.
    Output: the first several early-play decisions match the collaborator
    `RuleBasedPlayerV2` for the same public state.
- `test_ruleexact_configures_go_rule_2_prior`
    Input: no external input; construct the player.
    Output: the hybrid strategy is configured to use `go_rule_2`.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_ROOT = REPO_ROOT / "evaluate" / "GO-MCTS"
COLLAB_ROOT = REPO_ROOT / "Spades_AI_GO-MCTS"
for path in (REPO_ROOT, GO_MCTS_ROOT, COLLAB_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

bridge_path = GO_MCTS_ROOT / "bridge.py"
bridge_spec = importlib.util.spec_from_file_location("test_go_bridge", bridge_path)
if bridge_spec is None or bridge_spec.loader is None:
    raise RuntimeError(f"Unable to load bridge module from {bridge_path}")
bridge_module = importlib.util.module_from_spec(bridge_spec)
bridge_spec.loader.exec_module(bridge_module)

to_go_state = bridge_module.to_go_state
to_local_card = bridge_module.to_local_card

from strategy.rule_exact_player import RuleExactPlayer
from strategy.spades_match_runner import SpadesMatchRunner
from strategy.spades_player_programs import RandomSpadesPlayer
from trick_taking.card import Card, Rank, Suit
from trick_taking.game_state import GameState, Phase
from trick_taking.games.spades import SpadesRules

from spades_ai.game.scoring import BidType as GoBidType
from spades_ai.game.state import Bid as GoBid
from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as RuleBasedPlayerV2
from evaluate.evaluate_model_matchups import build_runtime, parse_args


def _advance_one_card(state: GameState, rules: SpadesRules, card: Card) -> None:
    current = state.turn
    state.play_card_to_table(current, card)
    if card.suit == state.trump_suit:
        state.trump_broken = True
        state.spades_broken = True

    if len(state.table_cards) == state.num_players:
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.turn = winner
        state.trick_leader = winner
    else:
        state.turn = (current + 1) % state.num_players


def _build_play_state(seed: int) -> tuple[GameState, SpadesRules]:
    runner = SpadesMatchRunner(
        [
            RandomSpadesPlayer(seed=seed + 0),
            RandomSpadesPlayer(seed=seed + 1),
            RandomSpadesPlayer(seed=seed + 2),
            RandomSpadesPlayer(seed=seed + 3),
        ],
        seed=seed,
        verbose=False,
    )
    runner._start_game()
    runner._bidding_phase()
    runner._set_teams()
    runner.state.phase = Phase.PLAYING
    return runner.state, runner.rules


def _build_bidding_state(seed: int) -> tuple[GameState, SpadesRules]:
    runner = SpadesMatchRunner(
        [
            RandomSpadesPlayer(seed=seed + 0),
            RandomSpadesPlayer(seed=seed + 1),
            RandomSpadesPlayer(seed=seed + 2),
            RandomSpadesPlayer(seed=seed + 3),
        ],
        seed=seed,
        verbose=False,
    )
    runner._start_game()
    return runner.state, runner.rules


def test_ruleexact_configures_go_rule_2_prior() -> None:
    player = RuleExactPlayer()
    assert player.strategy.config.prior_oracle_spec == "go_rule_2"
    assert player.strategy._prior_oracle is not None
    assert player.strategy.config.bid_checkpoint_path.endswith("bid_nsfp.pt")
    assert player._prior_oracle is not None
    assert player._bridge_mod is not None


def test_ruleexact_place_bid_uses_bid_mlp_result() -> None:
    state, rules = _build_bidding_state(seed=12345)
    player = RuleExactPlayer()

    class _StubBidPlayer:
        def choose_bid(self, _go_state):
            return GoBid(value=5, bid_type=GoBidType.NORMAL)

    player._bid_player = _StubBidPlayer()

    current = state.current_bidder
    legal_bids = rules.legal_bids(state, current)
    view = state.get_player_view(current)
    view["state"] = copy.deepcopy(state)

    chosen = player.place_bid(list(legal_bids), view)
    assert chosen == "bid_5"


def test_ruleexact_matches_rule_based_v2_in_early_play() -> None:
    state, rules = _build_play_state(seed=12345)
    ruleexact = RuleExactPlayer()
    rulebased_v2 = RuleBasedPlayerV2()

    # Check several early decisions while the remaining-card count is still > 28.
    for _ in range(8):
        remaining = sum(len(hand) for hand in state.hands)
        if remaining <= 28:
            break

        current = state.turn
        legal_cards = rules.playable(state, state.hands[current], current)
        assert legal_cards, "Expected at least one legal card during early play"

        view = state.get_player_view(current)
        view["state"] = copy.deepcopy(state)

        ours = ruleexact.play_card(list(legal_cards), view)
        go_state = to_go_state(copy.deepcopy(state))
        expected_go_card = rulebased_v2.choose_card(go_state)
        expected = to_local_card(expected_go_card)

        assert any(candidate.card_id == expected.card_id for candidate in legal_cards), (
            "Collaborator rule-based-v2 returned an illegal card"
        )
        assert ours.card_id == expected.card_id

        _advance_one_card(state, rules, ours)


def test_ruleexact_uses_rule_based_v2_for_remaining_25(monkeypatch) -> None:
    player = RuleExactPlayer()

    # Build a minimal state-like object: only `hands` is used in RuleExactPlayer.
    hands = [[object() for _ in range(7)], [object() for _ in range(6)], [object() for _ in range(6)], [object() for _ in range(6)]]
    state = SimpleNamespace(hands=hands)

    # Two legal cards; oracle will choose the second one.
    legal_cards = [
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
    ]

    class _StubOracle:
        def choose_card(self, _go_state):
            return object()

    class _StubBridge:
        @staticmethod
        def to_go_state(_state):
            return object()

        @staticmethod
        def to_local_card(_go_card):
            return SimpleNamespace(card_id=legal_cards[1].card_id)

    player._prior_oracle = _StubOracle()
    player._bridge_mod = _StubBridge()

    def _forbidden_choose_action(_state):
        raise AssertionError("choose_action should not be called when remaining >= 25")

    player.strategy.choose_action = _forbidden_choose_action
    chosen = player.play_card(legal_cards, {"state": state})
    assert chosen.card_id == legal_cards[1].card_id


def test_ruleexact_uses_strategy_for_remaining_24() -> None:
    player = RuleExactPlayer()

    hands = [[object() for _ in range(6)] for _ in range(4)]  # total = 24
    state = SimpleNamespace(hands=hands)
    legal_cards = [Card(Suit.SPADES, Rank.FOUR), Card(Suit.SPADES, Rank.FIVE)]

    # Even if oracle exists, <=24 must go through strategy.
    player._prior_oracle = object()
    player._bridge_mod = object()

    def _choose_action(_state):
        return legal_cards[0]

    player.strategy.choose_action = _choose_action
    chosen = player.play_card(legal_cards, {"state": state})
    assert chosen.card_id == legal_cards[0].card_id


def test_runtime_wires_exact_solver_count_from_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_model_matchups.py",
            "--our-number-of-exact-solvers",
            "137",
            "--num-games",
            "1",
        ],
    )
    args = parse_args()
    runtime = build_runtime(args)
    assert runtime.local_mcts_config.determinization_count == 137


def test_parse_args_defaults_disable_prior_oracle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_model_matchups.py"])
    args = parse_args()
    assert args.our_prior_oracle_spec == "no"


def test_import_failure_fallback_prints_at_most_five_times() -> None:
    from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy

    cfg = TruncatedMCTSConfig(prior_oracle_spec="go_rule_2")
    strategy = TruncatedMCTSStrategy(cfg)
    strategy._prior_oracle = None
    strategy._bridge_mod = None
    strategy._fallback_print_count = 0

    card = Card(Suit.HEARTS, Rank.TWO)
    initial_hands = [[card], [], [], []]
    play_sequence = [(0, card)]

    with patch("builtins.print") as mock_print:
        for _ in range(10):
            weight = strategy._compute_importance_weight(initial_hands, play_sequence, step_contexts=None)
            assert weight == 1.0

    assert mock_print.call_count == 5
