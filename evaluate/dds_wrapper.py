"""
DDS (Double Dummy Solver) ctypes wrapper for Spades.

Wraps the modified dds-bridge/dds library with spades_broken support.
Provides a Python interface for calling DDS's SolveBoard function.

Card encoding:
- Suits: 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs (matches local encoding)
- Ranks: bitmask where bit N = rank N present (bit 2=Two, ..., bit 14=Ace)
- remainCards[hand][suit] = 16-bit rank bitmask

Usage:
    solver = DDSSolver()
    results = solver.solve_position(hands, current_trick, trick_leader, next_to_play, spades_broken)
    # results: [{card: Card, tricks_won: int}, ...] sorted by tricks descending
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Any

from trick_taking.card import Card, Rank, Suit


# ─── Constants ────────────────────────────────────────────────────────────────
DDS_HANDS = 4
DDS_SUITS = 4
RETURN_NO_FAULT = 1


# ─── ctypes Structure Definitions ─────────────────────────────────────────────

class DDSDeal(ctypes.Structure):
    """Maps to DDS `Deal` struct (with spades_broken extension).

    Fields must match the C struct layout exactly:
      int trump;
      int first;
      int currentTrickSuit[3];
      int currentTrickRank[3];
      unsigned int remainCards[4][4];
      int spades_broken;
    """
    _fields_ = [
        ("trump", ctypes.c_int),
        ("first", ctypes.c_int),
        ("currentTrickSuit", ctypes.c_int * 3),
        ("currentTrickRank", ctypes.c_int * 3),
        ("remainCards", (ctypes.c_uint * DDS_SUITS) * DDS_HANDS),
        ("spades_broken", ctypes.c_int),
    ]


class DDSFutureTricks(ctypes.Structure):
    """Maps to DDS `FutureTricks` struct.

    Fields:
      int nodes;   - number of nodes searched
      int cards;   - number of distinct cards in result
      int suit[13]; - suit of each card option
      int rank[13]; - rank of each card option
      int equals[13]; - bitmask of equivalent lower-ranked cards
      int score[13]; - tricks won by side-to-play for each card
    """
    _fields_ = [
        ("nodes", ctypes.c_int),
        ("cards", ctypes.c_int),
        ("suit", ctypes.c_int * 13),
        ("rank", ctypes.c_int * 13),
        ("equals", ctypes.c_int * 13),
        ("score", ctypes.c_int * 13),
    ]


# ─── DDSSolver Wrapper Class ──────────────────────────────────────────────────

class DDSSolver:
    """High-level wrapper around the modified DDS shared library."""

    def __init__(self, lib_path: str | None = None):
        """Initialize the DDS solver.

        Args:
            lib_path: Path to the compiled libdds_spades.dylib.
                      If None, searches in standard locations.
        """
        if lib_path is None:
            lib_path = self._find_library()

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"DDS library not found: {lib_path}\n"
                f"Run: bash external/build_dds.sh"
            )

        self._lib = ctypes.CDLL(lib_path)

        # Set up SolveBoard signature
        # int SolveBoard(Deal dl, int target, int solutions, int mode,
        #                FutureTricks* futp, int threadIndex)
        self._lib.SolveBoard.argtypes = [
            DDSDeal,                            # deal (passed by value)
            ctypes.c_int,                       # target
            ctypes.c_int,                       # solutions
            ctypes.c_int,                       # mode
            ctypes.POINTER(DDSFutureTricks),    # result pointer
            ctypes.c_int,                       # threadIndex
        ]
        self._lib.SolveBoard.restype = ctypes.c_int

        # Initialize: set to single-threaded
        if hasattr(self._lib, 'SetMaxThreads'):
            self._lib.SetMaxThreads.argtypes = [ctypes.c_int]
            self._lib.SetMaxThreads.restype = ctypes.c_int
            self._lib.SetMaxThreads(1)

    @staticmethod
    def _find_library() -> str:
        """Search for the DDS library in standard locations."""
        candidates = [
            Path(__file__).resolve().parent.parent / "external" / "libdds_spades.dylib",
            Path(__file__).resolve().parent.parent / "external" / "libdds_spades.so",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return str(candidates[0])  # Return first candidate for error message

    def solve_position(
        self,
        hands: list[list[Card]],
        current_trick: list[tuple[int, Card]],
        trick_leader: int,
        next_to_play: int,
        spades_broken: bool,
    ) -> list[dict[str, Any]]:
        """Solve the position and return scored legal moves.

        Args:
            hands: hands[player_id] = list of Card objects in that player's hand
            current_trick: list of (player_id, Card) for cards already played
                          in the current trick, in play order
            trick_leader: player_id who leads this trick
            next_to_play: player_id whose turn it is now
            spades_broken: whether spades have been broken

        Returns:
            List of {card: Card, tricks_won: int} sorted by tricks_won descending.
            tricks_won is the number of tricks the side containing next_to_play
            can win from this position with perfect play.
        """
        deal = self._build_deal(
            hands, current_trick, trick_leader, next_to_play, spades_broken
        )

        fut = DDSFutureTricks()
        error_code = self._lib.SolveBoard(
            deal,
            -1,                 # target: find maximum tricks
            3,                  # solutions: return ALL moves with scores
            1,                  # mode: fresh transposition table
            ctypes.byref(fut),
            0,                  # threadIndex
        )

        if error_code != RETURN_NO_FAULT:
            raise RuntimeError(f"DDS SolveBoard failed with error code {error_code}")

        # Parse results
        results: list[dict[str, Any]] = []
        for i in range(fut.cards):
            suit_idx = fut.suit[i]
            rank_val = fut.rank[i]
            card = Card(Suit(suit_idx), Rank(rank_val))
            tricks = fut.score[i]
            results.append({"card": card, "tricks_won": tricks})

            # Expand "equals" bitmask for equivalent cards (same score)
            equals_mask = fut.equals[i]
            for r in range(2, 15):
                if equals_mask & (1 << r):
                    eq_card = Card(Suit(suit_idx), Rank(r))
                    results.append({"card": eq_card, "tricks_won": tricks})

        # Sort by tricks won (descending)
        results.sort(key=lambda x: x["tricks_won"], reverse=True)
        return results

    def get_best_card(
        self,
        hands: list[list[Card]],
        current_trick: list[tuple[int, Card]],
        trick_leader: int,
        next_to_play: int,
        spades_broken: bool,
    ) -> Card:
        """Return the single best card maximizing tricks for the side to play.

        Convenience wrapper around solve_position that returns just the best Card.
        """
        results = self.solve_position(
            hands, current_trick, trick_leader, next_to_play, spades_broken
        )
        if not results:
            raise RuntimeError("DDS returned no legal moves")
        return results[0]["card"]

    def _build_deal(
        self,
        hands: list[list[Card]],
        current_trick: list[tuple[int, Card]],
        trick_leader: int,
        next_to_play: int,
        spades_broken: bool,
    ) -> DDSDeal:
        """Convert game state to DDS Deal struct.

        Mapping: local seat 0-3 → DDS hand 0-3 (N, E, S, W).
        Teams: seats 0+2 vs 1+3 → N+S vs E+W (matches DDS partnership).
        """
        deal = DDSDeal()
        deal.trump = 0  # Always Spades (suit index 0)
        deal.first = trick_leader  # Who leads this trick
        deal.spades_broken = 1 if spades_broken else 0

        # Fill currentTrickSuit/Rank from cards already played in this trick
        # DDS expects cards in play order (first card at index 0)
        for i in range(3):
            if i < len(current_trick):
                _, card = current_trick[i]
                deal.currentTrickSuit[i] = card.suit.value
                deal.currentTrickRank[i] = card.rank.value
            else:
                deal.currentTrickSuit[i] = 0
                deal.currentTrickRank[i] = 0

        # Fill remainCards[hand][suit] = rank bitmask
        # Rank bitmask: bit N corresponds to rank N (bit 2 = Two, ..., bit 14 = Ace)
        for player_id in range(4):
            for s in range(4):
                deal.remainCards[player_id][s] = 0
            for card in hands[player_id]:
                deal.remainCards[player_id][card.suit.value] |= (1 << card.rank.value)

        return deal
