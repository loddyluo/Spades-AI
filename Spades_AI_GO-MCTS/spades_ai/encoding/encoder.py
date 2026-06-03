"""Observation encoder for the Spades AI.

Converts a GameState + perspective_player into a list[int] of token IDs
for use as input to an ML model.

Sequence format
---------------
[BOS]
  [POS_Px] [BID_<bid>]  × (number of bids placed so far)
  for each completed trick and the current in-progress trick:
    [STATE] <state-block tokens> [/STATE]
    [TRICK_SEP]
    [POS_Px] [C_<card>]  × (cards played in that trick)
[EOS]

The full sequence is capped at 256 tokens; a [PAD] token is *not* appended
here (the caller may pad).
"""
from __future__ import annotations

from spades_ai.encoding.vocabulary import Vocabulary
from spades_ai.game.card import Card, Rank, Suit
from spades_ai.game.scoring import BidType
from spades_ai.game.state import GameState, Phase
from spades_ai.game.trick import Trick, TrickCard

_MAX_LEN = 512

# Rank integer → character representation in card tokens
_RANK_TO_CHAR: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
    7: "7", 8: "8", 9: "9", 10: "10",
    11: "J", 12: "Q", 13: "K", 14: "A",
}

# Suit → lowercase char (card token suffix)
_SUIT_TO_CHAR: dict[Suit, str] = {
    Suit.CLUBS: "c",
    Suit.DIAMONDS: "d",
    Suit.HEARTS: "h",
    Suit.SPADES: "s",
}

# Suit → uppercase abbreviation (auxiliary tokens)
_SUIT_TO_ABBR: dict[Suit, str] = {
    Suit.CLUBS: "C",
    Suit.DIAMONDS: "D",
    Suit.HEARTS: "H",
    Suit.SPADES: "S",
}


def _card_token(card: Card) -> str:
    return f"C_{_RANK_TO_CHAR[card.rank.value]}{_SUIT_TO_CHAR[card.suit]}"


def _master_rank_in_suit(suit: Suit, played_cards: frozenset[Card]) -> str | None:
    """Return rank char of the current master card in *suit*, or None if suit depleted."""
    for rank_val in range(Rank.ACE, Rank.TWO - 1, -1):
        rank = Rank(rank_val)
        if Card(rank, suit) not in played_cards:
            return _RANK_TO_CHAR[rank_val]
    return None


def remaining_steps_in_state(state: GameState) -> int:
    """How many more card-play actions remain until end-of-hand.

    A full hand has 52 card-play actions (13 tricks * 4 cards).  We count
    cards already on the table — completed tricks plus the in-progress
    trick — and return the difference.  Used by the encoder to emit a
    ``STEPS_<n>`` token so the value head knows how far the game is from
    its end.
    """
    cards_played = (
        sum(len(t.cards) for t in state.completed_tricks)
        + len(state.current_trick_cards)
    )
    return max(0, 52 - cards_played)


class ObservationEncoder:
    """Encodes a GameState as a token-ID sequence from one player's perspective."""

    def __init__(self) -> None:
        self._vocab = Vocabulary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_policy_prefix(
        self,
        state: GameState,
        perspective_player: int,
    ) -> list[int]:
        """Return token IDs ending at ``POS_P<current_player>`` for policy inference.

        ``encode()`` returns a value-head sequence ending with
        ``[STEPS_<n>, EOS]``.  That suffix is correct for the value head but
        wrong for card-policy logits: during LM training, cards are predicted
        after a ``POS_Pi`` token inside a trick block:

            ``STATE ... TRICK_SEP POS_Pi -> C_card``

        This helper converts the value-head sequence into that policy prefix.
        If the current player is leading a fresh trick, ``encode()`` has no
        in-progress trick block yet, so we append the current ``STATE`` block
        plus ``TRICK_SEP`` before adding ``POS_P<current_player>``.
        """
        ids = self.encode(state, perspective_player)
        vocab = self._vocab

        if ids and ids[-1] == vocab.get_id("EOS"):
            ids = ids[:-1]
        if ids and vocab.id_to_token[ids[-1]].startswith("STEPS_"):
            ids = ids[:-1]

        if state.phase == Phase.PLAYING and not state.current_trick_cards:
            played_cards: frozenset[Card] = frozenset(
                tc.card
                for trick in state.completed_tricks
                for tc in trick.cards
            )
            current_state_tokens = self._state_block(
                state=state,
                trick_num=state.trick_number,
                perspective_player=perspective_player,
                played_cards=played_cards,
            )
            current_state_tokens.append("TRICK_SEP")
            ids.extend(
                vocab.get_id(tok)
                for tok in current_state_tokens
                if tok in vocab.token_to_id
            )

        ids.append(vocab.get_id(f"POS_P{state.current_player}"))
        if len(ids) > _MAX_LEN:
            ids = ids[-_MAX_LEN:]
        return ids

    def encode(
        self,
        state: GameState,
        perspective_player: int,
        *,
        remaining_steps: int | None = None,
    ) -> list[int]:
        """Return a list[int] of token IDs, length <= _MAX_LEN.

        ``remaining_steps`` controls the ``STEPS_<n>`` token appended just
        before EOS.  If ``None`` (default), it is computed from the state
        via :func:`remaining_steps_in_state` — this keeps the encoder
        backwards compatible for callers that don't pass the argument
        explicitly.  Search code that builds bare token streams (and so
        cannot recompute from a GameState) should pass it explicitly.
        """
        tokens: list[str] = []
        tokens.append("BOS")

        # ---------- Bid section ----------
        for i, bid in enumerate(state.bids):
            tokens.append(f"POS_P{i}")
            if bid.bid_type == BidType.NIL:
                tokens.append("BID_NIL")
            elif bid.bid_type == BidType.BLIND_NIL:
                tokens.append("BID_BNIL")
            else:
                tokens.append(f"BID_{bid.value}")

        # ---------- Tricks ----------
        # Collect all played cards to track master / suits remaining
        played_cards: frozenset[Card] = frozenset()

        # Completed tricks
        for trick_idx, trick in enumerate(state.completed_tricks):
            trick_num = trick_idx + 1
            state_tokens = self._state_block(
                state=state,
                trick_num=trick_num,
                perspective_player=perspective_player,
                played_cards=played_cards,
            )
            tokens += state_tokens
            tokens.append("TRICK_SEP")
            for tc in trick.cards:
                tokens.append(f"POS_P{tc.player}")
                tokens.append(_card_token(tc.card))
                played_cards = played_cards | {tc.card}

        # Current in-progress trick (if playing phase)
        if state.phase == Phase.PLAYING and state.current_trick_cards:
            trick_num = state.trick_number
            state_tokens = self._state_block(
                state=state,
                trick_num=trick_num,
                perspective_player=perspective_player,
                played_cards=played_cards,
            )
            tokens += state_tokens
            tokens.append("TRICK_SEP")
            for tc in state.current_trick_cards:
                tokens.append(f"POS_P{tc.player}")
                tokens.append(_card_token(tc.card))

        # ---------- Remaining-steps cue + EOS ----------
        if remaining_steps is None:
            remaining_steps = remaining_steps_in_state(state)
        rs = max(0, min(52, int(remaining_steps)))
        tokens.append(f"STEPS_{rs}")
        tokens.append("EOS")

        # Convert to IDs, truncating to MAX_LEN.  We keep the *tail* (not
        # the head) so that the trailing STEPS_<n> + EOS pair — which the
        # value head uses as its read anchor — is always preserved.  A
        # full 13-trick hand encodes to ~415 tokens, so MAX_LEN=512
        # comfortably fits everything; this branch is now defensive only.
        vocab = self._vocab
        ids = [vocab.get_id(t) for t in tokens if t in vocab.token_to_id]
        if len(ids) > _MAX_LEN:
            ids = ids[-_MAX_LEN:]
        return ids

    # ------------------------------------------------------------------
    # State block
    # ------------------------------------------------------------------

    def _state_block(
        self,
        state: GameState,
        trick_num: int,
        perspective_player: int,
        played_cards: frozenset[Card],
    ) -> list[str]:
        tokens: list[str] = ["STATE"]
        p = perspective_player
        partner = (p + 2) % 4
        opp1, opp2 = (p + 1) % 4, (p + 3) % 4

        # --- Trick number ---
        tokens.append(f"TRICK_NUM_{trick_num}")

        # --- Spades broken ---
        tokens.append("SPADES_BROKEN" if state.spades_broken else "SPADES_NOT_BROKEN")

        # --- My team contract status ---
        my_non_nil_bid = sum(
            state.bids[q].value
            for q in (p, partner)
            if state.bids[q].bid_type == BidType.NORMAL
        )
        my_tricks = state.tricks_won[p] + state.tricks_won[partner]
        needed = my_non_nil_bid - my_tricks
        if needed > 0:
            tokens.append(f"MY_NEED_{min(needed, 13)}")
        elif needed == 0:
            tokens.append("MY_NEED_0")
        else:
            tokens.append(f"MY_OVER_{min(-needed, 13)}")

        # --- Opponent team contract status ---
        opp_non_nil_bid = sum(
            state.bids[q].value
            for q in (opp1, opp2)
            if state.bids[q].bid_type == BidType.NORMAL
        )
        opp_tricks = state.tricks_won[opp1] + state.tricks_won[opp2]
        opp_needed = opp_non_nil_bid - opp_tricks
        if opp_needed > 0:
            tokens.append(f"OPP_NEED_{min(opp_needed, 13)}")
        elif opp_needed == 0:
            tokens.append("OPP_NEED_0")
        else:
            tokens.append(f"OPP_OVER_{min(-opp_needed, 13)}")

        # --- Nil / blind nil status for each player ---
        for q in range(4):
            bid = state.bids[q]
            if bid.bid_type == BidType.NIL:
                if state.tricks_won[q] == 0:
                    tokens.append(f"P{q}_NIL_ALIVE")
                else:
                    tokens.append(f"P{q}_NIL_BUSTED")
            elif bid.bid_type == BidType.BLIND_NIL:
                if state.tricks_won[q] == 0:
                    tokens.append(f"P{q}_BNIL_ALIVE")
                else:
                    tokens.append(f"P{q}_BNIL_BUSTED")

        # --- Void information ---
        for q in range(4):
            for suit in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS):
                if suit in state.void_shown[q]:
                    tokens.append(f"P{q}_VOID_{_SUIT_TO_ABBR[suit]}")

        # --- Suits remaining ---
        for suit in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS):
            abbr = _SUIT_TO_ABBR[suit]
            played_in_suit = sum(
                1 for c in played_cards if c.suit == suit
            )
            remaining = 13 - played_in_suit
            tokens.append(f"SUIT_LEFT_{abbr}_{remaining}")

        # --- Master cards ---
        for suit in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS):
            abbr = _SUIT_TO_ABBR[suit]
            rank_char = _master_rank_in_suit(suit, played_cards)
            if rank_char is not None:
                tokens.append(f"MASTER_{abbr}_{rank_char}")

        # --- Per-player tricks won ---
        for q in range(4):
            tokens.append(f"P{q}_WON_{state.tricks_won[q]}")

        # --- My spade count ---
        my_spade_ct = sum(1 for c in state.hands[p] if c.suit == Suit.SPADES)
        tokens.append(f"MY_SPADE_CT_{my_spade_ct}")

        # --- My hand size ---
        my_hand_size = len(state.hands[p])
        tokens.append(f"MY_HAND_SIZE_{my_hand_size}")

        tokens.append("/STATE")
        return tokens
