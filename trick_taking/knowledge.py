"""
Knowledge vectors for incomplete information tracking.

Paper reference: Section 3 "Data Structures" — "Knowledge Vectors"
"We use knowledge vectors k_{i,j} to encode the cards that a player P_j
knows that the other player P_i does not have, 1 ≤ i, j ≤ p. Each knowledge
vector k_{i,j} represents a set of cards."

"We use vectors of players not having a card, but one can also record cards
that a player has to have. The reason for using negation is that it is often
faster: if a player does not obey suit, then all cards of the suit can be
moved into the not-having vector."

The paper's update function (C++ pseudocode):
    void Player::updateBelief(Game* game, card c)
        for (int p = 0; p < N_PLAYERS; p++) have_not[p] |= c;
        for (card d : set_cards(~have_not[game.turn]))
            if ((c & playable(c | d, game)) == 0) have_not[game.turn] |= d;
"""

from __future__ import annotations

from trick_taking.card import Card, Suit, suit_mask, bitset_count


class KnowledgeTracker:
    """
    Tracks knowledge vectors k_{i,j} using bitsets.

    For each (observer, target) pair, we track which cards the target
    player is known NOT to have (`have_not` bitset). This is the paper's
    preferred representation because void-in-suit updates are fast:
    just OR the entire suit mask into have_not.

    Usage:
        tracker = KnowledgeTracker(num_players=4, deck_size=52)
        tracker.init_own_hand(player_id=0, hand_bitset=my_hand)
        tracker.card_played(player_id=2, card=some_card)
        tracker.mark_void(observer=0, target=2, suit=Suit.HEARTS)

        # Query: which cards might player 2 hold (from player 0's view)?
        possible = tracker.possible_cards(observer=0, target=2)
    """
    __slots__ = ("num_players", "deck_bitset", "_have_not")

    def __init__(self, num_players: int, all_cards_bitset: int) -> None:
        self.num_players = num_players
        self.deck_bitset = all_cards_bitset
        # have_not[observer][target] = bitset of cards target definitely doesn't have
        self._have_not: list[list[int]] = [
            [0] * num_players for _ in range(num_players)
        ]

    def init_own_hand(self, player_id: int, hand_bitset: int) -> None:
        """
        A player knows their own hand. Therefore, all OTHER players
        definitely don't have these cards (from this player's perspective).
        """
        for target in range(self.num_players):
            if target != player_id:
                self._have_not[player_id][target] |= hand_bitset

    def card_played(self, player_id: int, card: Card) -> None:
        """
        Paper: "for (int p = 0; p < N_PLAYERS; p++) have_not[p] |= c;"
        When a card is played, ALL players know NO ONE else has it.
        """
        bit = card.bit
        for observer in range(self.num_players):
            for target in range(self.num_players):
                self._have_not[observer][target] |= bit

    def mark_void(self, observer: int, target: int, suit: Suit) -> None:
        """
        Paper: "if a player does not obey suit, then all cards of the suit
        can be moved into the not-having vector."

        When observer sees that target didn't follow suit, mark ALL cards
        of that suit as "target doesn't have".
        """
        self._have_not[observer][target] |= suit_mask(suit)

    def possible_cards(self, observer: int, target: int) -> int:
        """
        Return bitset of cards that target MIGHT hold (from observer's view).
        This is the complement of have_not within the deck.
        """
        return self.deck_bitset & ~self._have_not[observer][target]

    def possible_count(self, observer: int, target: int) -> int:
        """Count of cards target might hold."""
        return bitset_count(self.possible_cards(observer, target))

    def update_void_from_play(self, observer: int, player_id: int,
                               card: Card, lead_suit: Suit) -> None:
        """
        If a player didn't follow the lead suit, they're void in it.
        This is the paper's inference step from cardPlayed.
        """
        if card.suit != lead_suit:
            self.mark_void(observer, player_id, lead_suit)
