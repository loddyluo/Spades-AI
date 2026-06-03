"""MLPBidPlayer: BidMLP-backed bidder with rule-based card play (Task 9)."""
from __future__ import annotations

import torch

from spades_ai.game.card import Card
from spades_ai.game.state import Bid, GameState
from spades_ai.game.scoring import BidType
from spades_ai.models.bid_encoder import BidEncoder
from spades_ai.players.rule_based.player import RuleBasedPlayer

# Index-to-Bid mapping for the 16 output slots of BidMLP
# Slots 0-13: normal bids (0 tricks through 13 tricks)
# Slot 14: Nil
# Slot 15: Blind Nil
_IDX_TO_BID: list[tuple[int, BidType]] = (
    [(i, BidType.NORMAL) for i in range(14)]
    + [(0, BidType.NIL), (0, BidType.BLIND_NIL)]
)


class MLPBidPlayer:
    """Uses a BidMLP for bidding and delegates card play to RuleBasedPlayer.

    Parameters
    ----------
    model:
        A trained BidMLP instance.
    device:
        Target device; auto-detected if None.
    """

    def __init__(self, model, device: str | None = None) -> None:
        self._model = model
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._encoder = BidEncoder()
        self._rule_player = RuleBasedPlayer()

    def choose_bid(self, state: GameState) -> Bid:
        """Choose a bid using the BidMLP."""
        player = state.current_player
        hand = list(state.hands[player])
        prev_bids = list(state.bids)
        # Position is how many bids have been placed so far (0, 1, 2, or 3)
        position = len(prev_bids)

        features = self._encoder.encode(hand, prev_bids, position)
        x = features.unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(x)  # (1, 16)

        idx = int(logits.argmax(dim=-1).item())
        value, bid_type = _IDX_TO_BID[idx]

        # Ensure normal bids are at least 1
        if bid_type == BidType.NORMAL and value < 1:
            value = 1

        return Bid(value=value, bid_type=bid_type)

    def choose_card(self, state: GameState) -> Card:
        """Delegate card selection to RuleBasedPlayer."""
        return self._rule_player.choose_card(state)
