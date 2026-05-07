"""
A Framework for General Trick-Taking Card Games

Based on: Edelkamp, S. "A Framework for General Trick-Taking Card Games"
KI 2024, LNAI 14992, pp. 73-85.

Architecture:
    - GameRules (ABC)  — Paper's Fig. 2 game configuration interface
    - AIPlayer (ABC)   — Paper's Section 4 AI player callback interface
    - GeneralCardGame  — Paper's Fig. 3 general driver loop
"""

from trick_taking.card import Card, Suit, Rank, RankOrder, STANDARD_RANK_ORDER
from trick_taking.deck import Deck, DeckConfig, STANDARD_52, SKAT_32, EUCHRE_24
from trick_taking.game_state import GameState
from trick_taking.game_rules import GameRules
from trick_taking.player import AIPlayer
from trick_taking.driver import GeneralCardGame, GameResult, MatchResult

__all__ = [
    "Card", "Suit", "Rank", "RankOrder", "STANDARD_RANK_ORDER",
    "Deck", "DeckConfig", "STANDARD_52", "SKAT_32", "EUCHRE_24",
    "GameState", "GameRules", "AIPlayer",
    "GeneralCardGame", "GameResult", "MatchResult",
]
