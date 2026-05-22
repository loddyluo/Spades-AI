"""
DDS-based perfect-information Spades player.

This "cheating" AI sees all 4 hands and uses the DDS double-dummy solver
to select the card that maximizes tricks won by its team.

Strategy:
- Bidding: Uses the shared MLP bid model (same as other players for fairness)
- Card play: Calls DDS with full state visibility to pick optimal card
- Scoring: Maximizes tricks (does NOT account for bags/overtricks)

This is intended as an upper-bound benchmark for evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trick_taking.card import Card, Suit
from trick_taking.player import AIPlayer
from evaluate.dds_wrapper import DDSSolver


class DDSPlayer(AIPlayer):
    """Perfect-information player using DDS for card play.

    Sees all 4 hands and uses double-dummy analysis to find the
    card that maximizes the number of tricks won by its side.

    Bidding uses the shared MLP model for fairness (so only card-play
    ability differs between teams).
    """

    def __init__(self, bid_model=None, bid_device: str = "cpu"):
        """
        Args:
            bid_model: Optional BidMLP model for bidding. If None, uses
                       simple hand-strength heuristic.
            bid_device: Torch device for bid model inference.
        """
        self._solver = DDSSolver()
        self._bid_model = bid_model
        self._bid_device = bid_device
        self.position: int = -1
        self.hand: list[Card] = []
        self.last_play_info: dict[str, Any] | None = None
        self.last_bid_info: dict[str, Any] | None = None

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self.last_play_info = None
        self.last_bid_info = None

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """Bid using MLP model (same model as opponents for fairness)."""
        if self._bid_model is not None:
            try:
                go_mcts_dir = REPO_ROOT / "evaluate" / "GO-MCTS"
                if str(go_mcts_dir) not in sys.path:
                    sys.path.insert(0, str(go_mcts_dir))

                from bridge import normalize_bid_for_legal_options, to_go_state
                from models import MLPBidPlayer

                state = state_view.get("state")
                if state is None:
                    return legal_bids[0]

                mlp_bidder = MLPBidPlayer(self._bid_model, self._bid_device)
                go_state = to_go_state(state)
                raw_bid = mlp_bidder.choose_bid(go_state)
                normalized = normalize_bid_for_legal_options(raw_bid, legal_bids)
                self.last_bid_info = {
                    "chosen_bid": normalized,
                    "legal_bids": list(legal_bids),
                }
                return normalized
            except Exception:
                pass

        # Fallback: simple hand-strength heuristic
        return self._simple_bid(legal_bids, state_view)

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        """Use DDS to pick the optimal card from the full-information state."""
        state = state_view.get("state")
        if state is None:
            self.last_play_info = {"mode": "no_state_fallback"}
            return legal_cards[0]

        # Single legal card → play it immediately
        if len(legal_cards) == 1:
            self.last_play_info = {"mode": "single_action", "best_value": None}
            return legal_cards[0]

        try:
            results = self._solver.solve_position(
                hands=[list(h) for h in state.hands],
                current_trick=list(state.table_cards),
                trick_leader=state.trick_leader,
                next_to_play=self.position,
                spades_broken=bool(state.spades_broken),
            )

            # Find best card that is also in legal_cards
            for result in results:
                candidate = result["card"]
                for legal_card in legal_cards:
                    if (legal_card.suit == candidate.suit and
                            legal_card.rank == candidate.rank):
                        self.last_play_info = {
                            "mode": "dds_exact",
                            "best_value": result["tricks_won"],
                            "action_scores": [
                                {"action": str(r["card"]), "value": r["tricks_won"]}
                                for r in results[:8]
                            ],
                        }
                        return legal_card

            # DDS result doesn't match legal cards (shouldn't happen)
            self.last_play_info = {"mode": "dds_no_match_fallback"}
            return legal_cards[0]

        except Exception as e:
            # DDS failure → graceful fallback
            self.last_play_info = {"mode": "dds_error", "error": str(e)}
            return self._greedy_fallback(legal_cards, state_view)

    def bid_placed(self, bidder: int, bid: Any) -> None:
        pass

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        pass

    def card_played(self, player_id: int, card: Card) -> None:
        pass

    def _simple_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """Fallback: estimate bid from high-card points."""
        state = state_view.get("state")
        if state is None or self.position < 0:
            return legal_bids[0] if legal_bids else None

        hand = state.hands[self.position]
        # Simple HCP: A=4, K=3, Q=2, J=1
        hcp = sum(max(0, card.rank.value - 10) for card in hand)
        # Spade length bonus
        spade_count = sum(1 for c in hand if c.suit == Suit.SPADES)
        estimated_tricks = hcp // 3 + max(0, spade_count - 3)
        estimated_tricks = max(1, min(13, estimated_tricks))

        target_bid = f"bid_{estimated_tricks}"
        if target_bid in legal_bids:
            self.last_bid_info = {"chosen_bid": target_bid}
            return target_bid

        # Find closest legal bid
        for bid in legal_bids:
            if isinstance(bid, str) and bid.startswith("bid_"):
                self.last_bid_info = {"chosen_bid": bid}
                return bid
        return legal_bids[0] if legal_bids else None

    def _greedy_fallback(self, legal_cards: list[Card], state_view: dict) -> Card:
        """Simple heuristic fallback when DDS fails."""
        # Play lowest non-trump card, or lowest trump
        non_trump = [c for c in legal_cards if c.suit != Suit.SPADES]
        if non_trump:
            return min(non_trump, key=lambda c: c.rank.value)
        return min(legal_cards, key=lambda c: c.rank.value)
