"""RuleExactPlayer: early-rule, late-exact hybrid player.

File purpose:
- Provide `RuleExactPlayer`, an `AIPlayer` that uses the collaborator's
  `rule_based_v2` for early-game decisions (when many cards remain) and
  falls back to the local `TruncatedMCTSStrategy` (which uses
  determinization + exact solver at leaf) for late-game decisions.

Functions / classes and I/O:
- `RuleExactPlayer(config: TruncatedMCTSConfig | None)`
  - Inputs: optional `TruncatedMCTSConfig` controlling exact/determinize behavior.
  - Methods:
    - `start_game(position:int, hand:list[Card], num_players:int) -> None`
    - `place_bid(legal_bids, state_view) -> Any` (random baseline)
     - `play_card(legal_cards, state_view) -> Card` uses collaborator rule when
         remaining_cards >= 25, otherwise delegates to
         `TruncatedMCTSStrategy.choose_action` (which runs exact+IS at <=24).

Notes:
- This module mirrors the bridge-loading approach used in
  `truncated_mcts_strategy` to convert local `GameState` to the
  collaborator `GoGameState` expected by `rule_based_v2`.
"""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from typing import Any

from trick_taking.card import Card
from trick_taking.player import AIPlayer
from trick_taking.game_state import GameState

from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy


class RuleExactPlayer(AIPlayer):
    """Hybrid player: early use rule_based_v2; late delegate to TruncatedMCTSStrategy."""

    def __init__(self, config: TruncatedMCTSConfig | None = None) -> None:
        self.position = -1
        self.hand: list[Card] = []
        self.config = config or TruncatedMCTSConfig()
        if self.config.prior_oracle_spec in {"", "no"}:
            self.config.prior_oracle_spec = "go_rule_2"
        if not self.config.bid_checkpoint_path:
            default_bid = Path(__file__).resolve().parents[1] / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_nsfp.pt"
            self.config.bid_checkpoint_path = str(default_bid)
        self.strategy = TruncatedMCTSStrategy(self.config)
        self._rng = random.Random()

        # Try to load collaborator rule_based_v2 and bridge if available.
        self._prior_oracle = None
        self._bridge_mod = None
        try:
            from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as _RBP  # type: ignore
            self._prior_oracle = _RBP()
        except Exception:
            self._prior_oracle = None

        self._bid_player = None
        if self.config.bid_checkpoint_path:
            try:
                import torch
                from spades_ai.models.bid_mlp import BidMLP  # type: ignore
                from spades_ai.players.mlp_bid_player import MLPBidPlayer  # type: ignore

                if os.path.exists(self.config.bid_checkpoint_path):
                    bid_model = BidMLP()
                    state_dict = torch.load(self.config.bid_checkpoint_path, weights_only=True, map_location="cpu")
                    bid_model.load_state_dict(state_dict)
                    bid_model.eval()
                    self._bid_player = MLPBidPlayer(bid_model, device="cpu")
            except Exception:
                self._bid_player = None

        # Load bridge.py from evaluate/GO-MCTS if present
        try:
            base = os.path.dirname(__file__)
            bridge_path = os.path.normpath(os.path.join(base, "..", "evaluate", "GO-MCTS", "bridge.py"))
            if os.path.exists(bridge_path):
                spec = importlib.util.spec_from_file_location("_go_bridge", bridge_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._bridge_mod = mod
        except Exception:
            self._bridge_mod = None

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        state: GameState | None = state_view.get("state")
        if state is None:
            raise ValueError("RuleExactPlayer.place_bid requires state_view['state']")
        if self._bridge_mod is None:
            raise RuntimeError("RuleExactPlayer requires bridge.py for bid normalization")
        if self._bid_player is None:
            raise RuntimeError(
                f"RuleExactPlayer requires bid_nsfp checkpoint at {self.config.bid_checkpoint_path!r}"
            )

        go_state = self._bridge_mod.to_go_state(state)
        raw_bid = self._bid_player.choose_bid(go_state)
        return self._bridge_mod.normalize_bid_for_legal_options(raw_bid, legal_bids)

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        state: GameState | None = state_view.get("state")
        if state is None:
            raise ValueError("RuleExactPlayer.play_card requires state_view['state']")

        remaining = sum(len(h) for h in state.hands)

        # Early/mid game: remaining 52..25 cards -> deterministic rule_based_v2.
        if remaining >= 25:
            if self._prior_oracle is None or self._bridge_mod is None:
                raise RuntimeError("RuleExactPlayer requires collaborator rule_based_v2 and bridge for remaining_cards >= 25")
            try:
                go_state = self._bridge_mod.to_go_state(state)
                go_card = self._prior_oracle.choose_card(go_state)
                local_card = self._bridge_mod.to_local_card(go_card)
                # Ensure selected card is legal; return the canonical object from
                # `legal_cards` (engine expects one of those instances).
                for c in legal_cards:
                    if c.card_id == local_card.card_id:
                        return c
                raise RuntimeError(f"rule_based_v2 returned illegal card: {local_card!r}")
            except Exception:
                raise

        # Endgame (<=24 expected by config): use strategy path that triggers
        # determinization + importance sampling + exact solver.
        action = self.strategy.choose_action(state)
        if action is None:
            # As fallback pick random legal
            if not legal_cards:
                raise ValueError("No legal cards available")
            return self._rng.choice(legal_cards)
        return action
