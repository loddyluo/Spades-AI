"""BidEncoder: encodes a player's hand, previous bids, and seat position
into a fixed 149-dimensional float tensor.

Layout (149 dimensions):
  [0:52]   hand one-hot  (4 suits × 13 ranks = 52)
  [52:116] bids one-hot  (4 players × 16 slots = 64)
  [116:119] position one-hot (3 slots; seat 3 is handled by the caller as
             positions 0-2 relative to the bidding order)
  [119:149] derived features (30, padded with zeros)

Bid index mapping (per player, 16 slots):
  0-13  normal bids (0 through 13 tricks)
  14    Nil
  15    Blind Nil

Derived features (30 total, zeros past the 16 meaningful values):
  [0:4]  suit lengths (clubs, diamonds, hearts, spades)
  [4:8]  high card count per suit (A/K/Q = "high" cards, raw count)
  [8:12] ace count per suit (0 or 1 each)
  [12]   spade power (spades/13)
  [13]   void count (# suits with 0 cards)
  [14]   partner bid (-1 if unknown, else bid index / 15)
  [15]   opponent total bid (-1 if unknown, else total / 26)
  [16:30] zero padding
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from spades_ai.game.card import Card, Suit, Rank
from spades_ai.game.state import Bid, BidType

_SUITS = list(Suit)   # [CLUBS, DIAMONDS, HEARTS, SPADES]
_RANKS = list(Rank)   # [TWO, …, ACE] (13 items)
_HIGH_RANKS = {Rank.ACE, Rank.KING, Rank.QUEEN}

_HAND_DIM = 52      # 4*13
_BID_DIM = 64       # 4*16
_POS_DIM = 3
_DERIVED_DIM = 30
INPUT_DIM = _HAND_DIM + _BID_DIM + _POS_DIM + _DERIVED_DIM  # 149
_SPADES_SLOT = _SUITS.index(Suit.SPADES)  # spade power reads this suit block


def _bid_index(bid: Bid) -> int:
    if bid.bid_type == BidType.NIL:
        return 14
    if bid.bid_type == BidType.BLIND_NIL:
        return 15
    return bid.value  # 0-13


class BidEncoder:
    """Encodes bidding-phase information into a 149-d float32 tensor."""

    @property
    def input_dim(self) -> int:
        return INPUT_DIM

    def encode(
        self,
        hand: list[Card],
        prev_bids: list[Bid],
        position: int,
    ) -> torch.Tensor:
        """Encode a single (hand, bids, position) triple.

        Args:
            hand:       The cards in the player's hand.
            prev_bids:  Bids placed so far, in seat order starting from seat 0.
            position:   This player's position relative to the bidding order (0-2).
                        Seat 3 is treated as position 3 but we one-hot encode
                        using clamp so values beyond 2 fall outside the 3-slot
                        block (callers should pass 0-2).

        Returns:
            float32 tensor of shape (149,).
        """
        vec = torch.zeros(INPUT_DIM, dtype=torch.float32)

        # ── hand one-hot [0:52] ──────────────────────────────────────────
        for card in hand:
            vec[card.index] = 1.0

        # ── bids one-hot [52:116] ────────────────────────────────────────
        for player_idx, bid in enumerate(prev_bids):
            if player_idx >= 4:
                break
            slot = 52 + player_idx * 16 + _bid_index(bid)
            vec[slot] = 1.0

        # ── position one-hot [116:119] ───────────────────────────────────
        if 0 <= position < _POS_DIM:
            vec[116 + position] = 1.0

        # ── derived features [119:149] ───────────────────────────────────
        vec[119:149] = _derived_features(hand, prev_bids)

        return vec

    def encode_indices_batch(
        self,
        hand_indices: Sequence[Sequence[int]],
        bid_slots: Sequence[Sequence[int]],
        positions: Sequence[int],
    ) -> torch.Tensor:
        """Vectorised :meth:`encode` for a whole batch at once.

        ``encode`` builds each row with ~25 tiny torch ops (a ``zeros`` fill,
        then one scalar assignment per card, per bid, per position, plus the 30
        derived slots), which costs far more in dispatch than in arithmetic.
        This twin computes the identical 149-vector per row with numpy and
        converts once, so the cost stops being proportional to the row count.

        Args:
            hand_indices: per row, the 0..51 card indices in the player's hand.
            bid_slots:    per row, the four seats' bid slot indices (0..15); a
                          negative entry means that seat has not bid yet.
            positions:    per row, the relative bidding position.

        Returns:
            float32 tensor of shape (N, 149), bit-identical to stacking
            :meth:`encode` over the same rows.
        """

        rows = len(positions)
        if len(hand_indices) != rows or len(bid_slots) != rows:
            raise ValueError("hand, bid and position batches must be the same length")

        feats = np.zeros((rows, INPUT_DIM), dtype=np.float64)

        for row, indices in enumerate(hand_indices):
            ids = np.asarray(indices, dtype=np.int64)
            if ids.size:
                if ids.min() < 0 or ids.max() >= _HAND_DIM:
                    raise ValueError("hand indices must be in [0, 51]")
                feats[row, ids] = 1.0

        filled_slots = np.full((rows, 4), -1, dtype=np.int64)
        for row, slots in enumerate(bid_slots):
            for seat in range(min(4, len(slots))):
                slot = int(slots[seat])
                if slot < 0:
                    continue
                if slot >= 16:
                    raise ValueError("bid slots must be in [0, 15]")
                filled_slots[row, seat] = slot
                feats[row, _HAND_DIM + seat * 16 + slot] = 1.0

        for row, position in enumerate(positions):
            position = int(position)
            if 0 <= position < _POS_DIM:
                feats[row, _HAND_DIM + _BID_DIM + position] = 1.0

        # Derived block, read back out of the hand one-hot block so the suit
        # layout can never drift from the one above.
        suit_bases = np.arange(4) * 13
        suit_lengths = np.stack(
            [feats[:, base : base + 13].sum(axis=1) for base in suit_bases],
            axis=1,
        )
        high_counts = np.stack(
            [feats[:, base + 10 : base + 13].sum(axis=1) for base in suit_bases],
            axis=1,
        )
        ace_counts = np.stack(
            [feats[:, base + 12] for base in suit_bases],
            axis=1,
        )

        derived = np.zeros((rows, _DERIVED_DIM), dtype=np.float64)
        derived[:, 0:4] = suit_lengths / 13.0
        derived[:, 4:8] = high_counts / 3.0
        derived[:, 8:12] = ace_counts
        derived[:, 12] = suit_lengths[:, _SPADES_SLOT] / 13.0
        derived[:, 13] = (suit_lengths == 0).sum(axis=1) / 4.0

        partner = filled_slots[:, 2]
        derived[:, 14] = np.where(partner >= 0, partner / 15.0, -1.0)
        opponent_known = (filled_slots[:, 1] >= 0) & (filled_slots[:, 3] >= 0)
        opponent_total = (
            np.where(filled_slots[:, 1] >= 0, filled_slots[:, 1], 0)
            + np.where(filled_slots[:, 3] >= 0, filled_slots[:, 3], 0)
        )
        derived[:, 15] = np.where(opponent_known, opponent_total / 26.0, -1.0)

        feats[:, _HAND_DIM + _BID_DIM + _POS_DIM :] = derived
        return torch.from_numpy(feats.astype(np.float32))

    def batch_encode(
        self,
        items: Iterable[tuple[list[Card], list[Bid], int]],
    ) -> torch.Tensor:
        """Encode a batch of (hand, prev_bids, position) triples.

        Returns:
            float32 tensor of shape (N, 149).
        """
        hands = []
        slots = []
        positions = []
        for hand, prev_bids, position in items:
            hands.append([card.index for card in hand])
            row = [_bid_index(bid) for bid in prev_bids[:4]]
            row.extend([-1] * (4 - len(row)))
            slots.append(row)
            positions.append(position)
        return self.encode_indices_batch(hands, slots, positions)


# ── helper ────────────────────────────────────────────────────────────────────

def _derived_features(hand: list[Card], prev_bids: list[Bid]) -> torch.Tensor:
    """Compute the 30-d derived feature vector."""
    feat = torch.zeros(30, dtype=torch.float32)

    # Suit lengths [0:4]
    suit_cards: dict[Suit, list[Card]] = {s: [] for s in _SUITS}
    for card in hand:
        suit_cards[card.suit].append(card)

    for i, suit in enumerate(_SUITS):
        feat[i] = len(suit_cards[suit]) / 13.0

    # High card counts per suit [4:8]  (A/K/Q)
    for i, suit in enumerate(_SUITS):
        high = sum(1 for c in suit_cards[suit] if c.rank in _HIGH_RANKS)
        feat[4 + i] = high / 3.0

    # Ace count per suit [8:12]
    for i, suit in enumerate(_SUITS):
        has_ace = any(c.rank == Rank.ACE for c in suit_cards[suit])
        feat[8 + i] = 1.0 if has_ace else 0.0

    # Spade power [12]
    feat[12] = len(suit_cards[Suit.SPADES]) / 13.0

    # Void count [13]
    void_count = sum(1 for s in _SUITS if len(suit_cards[s]) == 0)
    feat[13] = void_count / 4.0

    # Partner bid [14]  (partner is at offset +2; unknown if not yet placed)
    # prev_bids[2] would be player 2's bid if this player is 0
    if len(prev_bids) >= 3:
        feat[14] = _bid_index(prev_bids[2]) / 15.0
    else:
        feat[14] = -1.0

    # Opponent total bid [15]  (opponents are at indices 1 and 3)
    opp_known = True
    opp_total = 0
    for opp_idx in (1, 3):
        if len(prev_bids) > opp_idx:
            opp_total += _bid_index(prev_bids[opp_idx])
        else:
            opp_known = False
    feat[15] = opp_total / 26.0 if opp_known else -1.0

    # [16:30] remain zero (padding)
    return feat
