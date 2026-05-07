"""
AI Player abstract base class — the paper's Section 4 interface.

Paper reference: Section 4 "General Card Game Play"
"The AI player interface is a set of (virtual abstract) functions and looks
like follows. All the players operate fully independent of each other, and
have no other information than provided by the server alias driver loop,
who also receives all the player's answers."

Each concrete player (RandomPlayer, HumanPlayer, PIMC, αμ, etc.) implements
this ABC. The driver loop calls these callbacks during game phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trick_taking.card import Card


class AIPlayer(ABC):
    """
    Abstract AI player interface (Paper's Section 4 callbacks).

    The driver loop (Fig. 3) calls these methods at specific game events.
    Players ONLY receive information through these callbacks — they do NOT
    have direct access to the GameState.
    """

    # ─── Game start ──────────────────────────────────────────────────

    @abstractmethod
    def start_game(self, position: int, hand: list[Card],
                   num_players: int) -> None:
        """
        Paper: "startGame — Called when a new game is started.
        Parameters: position at table and hand cards array with card codes."

        Called once per deal. The player should store its position and hand.
        """
        ...

    # ─── Bidding phase ───────────────────────────────────────────────

    def place_bid(self, legal_bids: list[Any],
                  state_view: dict) -> Any:
        """
        Paper: "placeBid — Called when AI has to confirm a bid.
        Parameter: bidding value to increase."

        Called when it's this player's turn to bid.
        Must return one of the legal_bids.
        Default: pass or minimum bid.
        """
        return legal_bids[0] if legal_bids else None

    def bid_placed(self, player_id: int, bid_value: Any) -> None:
        """
        Paper: "bidPlaced — Announcing bid to other players.
        Parameters: bidding value."

        Called for ALL players when any player places a bid.
        """
        pass

    # ─── Team assignment ─────────────────────────────────────────────

    def set_teams(self, teams: list[int], bids: list[Any]) -> None:
        """
        Paper: "setTeams — called when AI has lost the bidding.
        Parameters: team: Boolean array of players, array of highest bid."

        Called after bidding to inform all players of team assignments.
        """
        pass

    # ─── Declaration phase (Skat/Tarot) ──────────────────────────────

    def declare_game(self, legal_declarations: list[Any],
                     state_view: dict) -> Any:
        """
        Paper: "declareGame — Called when AI has requested has to declare a game.
        Parameters: game declaration record."

        Only called for games with a declaration phase.
        """
        return legal_declarations[0] if legal_declarations else None

    def is_hand_game(self, dog: list[Card],
                     state_view: dict) -> bool:
        """
        Paper: "isHandGame — Asking AI if it requests a handgame.
        Parameters: game declaration record."

        Only called for games with dog/skat exchange.
        Return True to play hand game (keep dog), False to exchange.
        """
        return False

    def set_game(self, declaration: Any) -> None:
        """
        Paper: "setGame — Information sent to every player after game is declared.
        Parameters: game declaration record."

        Called for ALL players after declaration is finalized.
        """
        pass

    # ─── Trick play ──────────────────────────────────────────────────

    @abstractmethod
    def play_card(self, legal_cards: list[Card],
                  state_view: dict) -> Card:
        """
        Paper: "playCard — called when AI has to play a card.
        No parameters."

        Called when it's this player's turn to play a card.
        Must return one of the legal_cards.
        """
        ...

    def card_played(self, player_id: int, card: Card) -> None:
        """
        Paper: "cardPlayed — called always when a card is played by a player.
        Parameters: position of player who played the card, card code of
        played card."

        Called for ALL players whenever any card is played.
        Players can use this to update their knowledge vectors.
        """
        pass
