"""Token vocabulary for the Spades AI observation encoder.

Tokens are grouped into structural, card, position, bid, and auxiliary
(Tier 1-3) categories.  Every token is assigned a unique integer ID.
"""
from __future__ import annotations

_RANK_CHARS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_SUIT_CHARS_CARD = ["c", "d", "h", "s"]   # for C_<rank><suit> tokens
_SUIT_ABBRS = ["C", "D", "H", "S"]        # for auxiliary tokens


def _build_token_list() -> list[str]:
    tokens: list[str] = []

    # ------------------------------------------------------------------ #
    # Structural (6)
    # ------------------------------------------------------------------ #
    tokens += ["BOS", "EOS", "PAD", "STATE", "/STATE", "TRICK_SEP"]

    # ------------------------------------------------------------------ #
    # Card tokens (52): C_2c, C_3c, ... C_As
    # Order: clubs (c), diamonds (d), hearts (h), spades (s)
    # ------------------------------------------------------------------ #
    for suit_char in _SUIT_CHARS_CARD:
        for rank_char in _RANK_CHARS:
            tokens.append(f"C_{rank_char}{suit_char}")

    # ------------------------------------------------------------------ #
    # Position tokens (4)
    # ------------------------------------------------------------------ #
    for i in range(4):
        tokens.append(f"POS_P{i}")

    # ------------------------------------------------------------------ #
    # Bid tokens (16): BID_0..13, BID_NIL, BID_BNIL
    # ------------------------------------------------------------------ #
    for i in range(14):
        tokens.append(f"BID_{i}")
    tokens += ["BID_NIL", "BID_BNIL"]

    # ------------------------------------------------------------------ #
    # Tier 1 auxiliary tokens
    # ------------------------------------------------------------------ #

    # MY_NEED_0..13, MY_OVER_0..13, MY_SET
    for i in range(14):
        tokens.append(f"MY_NEED_{i}")
    for i in range(14):
        tokens.append(f"MY_OVER_{i}")
    tokens.append("MY_SET")

    # OPP_NEED_0..13, OPP_OVER_0..13, OPP_SET
    for i in range(14):
        tokens.append(f"OPP_NEED_{i}")
    for i in range(14):
        tokens.append(f"OPP_OVER_{i}")
    tokens.append("OPP_SET")

    # Px_NIL_ALIVE / Px_NIL_BUSTED / Px_BNIL_ALIVE / Px_BNIL_BUSTED
    for p in range(4):
        tokens.append(f"P{p}_NIL_ALIVE")
        tokens.append(f"P{p}_NIL_BUSTED")
        tokens.append(f"P{p}_BNIL_ALIVE")
        tokens.append(f"P{p}_BNIL_BUSTED")

    # Px_VOID_S / _H / _D / _C
    for p in range(4):
        for suit in ("S", "H", "D", "C"):
            tokens.append(f"P{p}_VOID_{suit}")

    # ------------------------------------------------------------------ #
    # Tier 2 auxiliary tokens
    # ------------------------------------------------------------------ #

    # SUIT_LEFT_<suit>_0..13
    for suit in ("S", "H", "D", "C"):
        for i in range(14):
            tokens.append(f"SUIT_LEFT_{suit}_{i}")

    # MASTER_<suit>_<rank>
    for suit in ("S", "H", "D", "C"):
        for rank_char in _RANK_CHARS:
            tokens.append(f"MASTER_{suit}_{rank_char}")

    # TRICK_NUM_1..13
    for i in range(1, 14):
        tokens.append(f"TRICK_NUM_{i}")

    # TRICK_LEAD_<suit>
    for suit in ("S", "H", "D", "C"):
        tokens.append(f"TRICK_LEAD_{suit}")

    # TRICK_WINNING_P0..3
    for p in range(4):
        tokens.append(f"TRICK_WINNING_P{p}")

    tokens += ["I_AM_LEAD", "I_AM_LAST"]

    # ------------------------------------------------------------------ #
    # Tier 3 auxiliary tokens
    # ------------------------------------------------------------------ #

    # Px_WON_0..13
    for p in range(4):
        for i in range(14):
            tokens.append(f"P{p}_WON_{i}")

    tokens += ["SPADES_BROKEN", "SPADES_NOT_BROKEN"]

    # MY_SPADE_CT_0..13
    for i in range(14):
        tokens.append(f"MY_SPADE_CT_{i}")

    # MY_HAND_SIZE_0..13
    for i in range(14):
        tokens.append(f"MY_HAND_SIZE_{i}")

    # ------------------------------------------------------------------ #
    # Remaining-steps token (Tier 4 — added 2026-04-28)
    # ------------------------------------------------------------------ #
    # STEPS_<n> with n in [0, 52] tells the value head how many more
    # card-play actions remain until the hand ends.  This token is
    # appended right before EOS in encoder.encode(), so the value head
    # (anchored on EOS) attends to it via causal attention and learns to
    # condition its score-diff prediction on "how far we still are".
    for i in range(53):
        tokens.append(f"STEPS_{i}")

    return tokens


class Vocabulary:
    """Complete token vocabulary for the Spades observation encoder."""

    def __init__(self) -> None:
        token_list = _build_token_list()
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(token_list)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(token_list)}

    @property
    def size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.token_to_id)

    def get_id(self, token: str) -> int:
        """Return the integer ID for *token* (raises KeyError if unknown)."""
        return self.token_to_id[token]
