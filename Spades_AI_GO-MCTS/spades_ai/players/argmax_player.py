"""ArgmaxPlayer: GPT2-backed card player with rule-based bidding (Task 8)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from spades_ai.encoding.encoder import (
    ObservationEncoder,
    _RANK_TO_CHAR,
    _SUIT_TO_CHAR,
)
from spades_ai.game.card import Card
from spades_ai.game.legal_moves import get_legal_moves
from spades_ai.game.state import Bid, GameState
from spades_ai.players.rule_based.bidding import rule_based_bid


class ArgmaxPlayer:
    """Plays cards using argmax over GPT-2 policy logits; bids via rule-based logic.

    Card selection:
    1. Encode the current game state.
    2. Strip the value-head suffix ``[STEPS_<n>, EOS]``.
    3. Append ``POS_P<current_player>`` and read LM logits at that position.
    4. Map each legal card to its vocabulary token probability.
    5. Filter out cards below *threshold* (relative to max probability).
    6. Return the card with the highest probability.

    Parameters
    ----------
    model:
        A GPT2PolicyValueModel (eval mode is set internally).
    threshold:
        Minimum fraction of the best card probability required for a card to
        remain eligible. Cards below ``threshold * max_prob`` are filtered out.
    device:
        Target device; auto-detected if None.
    """

    def __init__(self, model, threshold: float = 0.05, device: str | None = None) -> None:
        self._model = model
        self._threshold = threshold
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._encoder = ObservationEncoder()

    def choose_bid(self, state: GameState) -> Bid:
        """Delegate bidding to the rule-based strategy."""
        player = state.current_player
        return rule_based_bid(state.hands[player], state.bids, player)

    def choose_card(self, state: GameState) -> Card:
        """Choose a card using the policy model LM logits."""
        player = state.current_player
        hand = state.hands[player]
        is_leading = len(state.current_trick_cards) == 0

        legal = get_legal_moves(
            hand=hand,
            led_suit=state.led_suit,
            spades_broken=state.spades_broken,
            is_leading=is_leading,
        )

        # Build the policy prefix ending at POS_P<current_player>.  This is
        # the position where the LM head was trained to predict the card.
        token_ids = self._encoder.encode_policy_prefix(state, player)
        if not token_ids:
            return next(iter(legal))

        vocab = self._encoder._vocab.token_to_id

        max_len = self._model.config.n_positions
        token_ids = token_ids[-max_len:]  # keep the most-recent tokens

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)

        with torch.no_grad():
            lm_logits, _ = self._model(input_ids)

        # Logits at the appended POS_P<player> position: shape (vocab_size,)
        last_logits = lm_logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        # Map each legal card to its probability via its token name
        card_probs: dict[Card, float] = {}

        for card in legal:
            token_name = f"C_{_RANK_TO_CHAR[card.rank.value]}{_SUIT_TO_CHAR[card.suit]}"
            if token_name in vocab:
                token_id = vocab[token_name]
                card_probs[card] = probs[token_id].item()
            else:
                card_probs[card] = 0.0

        if not card_probs:
            return next(iter(legal))

        max_prob = max(card_probs.values())

        # Filter by threshold relative to max probability
        if max_prob > 0:
            filtered = {
                c: p for c, p in card_probs.items()
                if p >= self._threshold * max_prob
            }
        else:
            filtered = card_probs

        if not filtered:
            filtered = card_probs

        # Argmax over filtered candidates
        return max(filtered, key=lambda c: filtered[c])
