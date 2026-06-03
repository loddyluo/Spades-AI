from __future__ import annotations
from dataclasses import dataclass
from spades_ai.game.card import Card, Suit


@dataclass(frozen=True)
class TrickCard:
    player: int
    card: Card


@dataclass(frozen=True)
class Trick:
    cards: tuple[TrickCard, ...]
    led_suit: Suit

    def winner(self) -> int:
        best_player = self.cards[0].player
        best_card = self.cards[0].card
        for tc in self.cards[1:]:
            if _beats(tc.card, best_card, self.led_suit):
                best_player = tc.player
                best_card = tc.card
        return best_player

    def contains_spade(self) -> bool:
        return any(tc.card.suit == Suit.SPADES for tc in self.cards)

    def winning_card(self) -> Card:
        winner_idx = self.winner()
        for tc in self.cards:
            if tc.player == winner_idx:
                return tc.card
        raise ValueError("Winner not found")


def _beats(challenger: Card, current_best: Card, led_suit: Suit) -> bool:
    c_trump = challenger.suit == Suit.SPADES
    b_trump = current_best.suit == Suit.SPADES
    if c_trump and not b_trump:
        return True
    if not c_trump and b_trump:
        return False
    if c_trump and b_trump:
        return challenger.rank > current_best.rank
    c_follows = challenger.suit == led_suit
    b_follows = current_best.suit == led_suit
    if c_follows and not b_follows:
        return True
    if not c_follows and b_follows:
        return False
    if c_follows and b_follows:
        return challenger.rank > current_best.rank
    return False
