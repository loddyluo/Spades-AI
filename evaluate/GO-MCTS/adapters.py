"""Player adapters for mixing collaborator models with the local runner.

File purpose:
- Wrap collaborator `spades_ai` players so they can be used by the local
  `trick_taking`-based match runner.
- Provide a local player that uses the collaborator hand-strength heuristic
  for bidding and the local truncated MCTS strategy for card play.

Function input/output summary:
- None of the public behavior is exposed as free functions; the file defines
  adapter classes whose methods are documented individually below.
"""

from __future__ import annotations

from typing import Any

from trick_taking.card import Card as LocalCard
from trick_taking.player import AIPlayer

from strategy.hand_strength import _hand_strength
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy

from bridge import normalize_bid_for_legal_options, to_go_state


class GoPlayerAdapter(AIPlayer):
    """Adapt a collaborator `spades_ai` player to the local `AIPlayer` API."""

    def __init__(self, go_player: Any) -> None:
        """Store the wrapped collaborator player.

        Input:
        - go_player: an object implementing `choose_bid(GameState)` and
          `choose_card(GameState)` from the collaborator repository.

        Output:
        - A local `AIPlayer`-compatible adapter.
        """
        self._go_player = go_player
        self.position = -1
        self.hand: list[LocalCard] = []

    def start_game(self, position: int, hand: list[LocalCard], num_players: int) -> None:
        """Store the local seat and hand for compatibility with the runner.

        Input:
        - position: local seat index.
        - hand: list of local `Card` objects at deal time.
        - num_players: number of players in the hand.

        Output:
        - None.
        """
        self.position = position
        self.hand = list(hand)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """Ask the wrapped collaborator player for a bid and normalize it.

        Input:
        - legal_bids: the exact legal bid list from the local rules engine.
        - state_view: local player view dictionary containing a full `state`.

        Output:
        - One of `legal_bids`.
        """
        state = state_view.get("state")
        if state is None:
            raise ValueError("GoPlayerAdapter.place_bid requires state_view['state']")
        go_state = to_go_state(state)
        raw_bid = self._go_player.choose_bid(go_state)
        return normalize_bid_for_legal_options(raw_bid, legal_bids)

    def play_card(self, legal_cards: list[LocalCard], state_view: dict) -> LocalCard:
        """Ask the wrapped collaborator player for a card and convert it back.

        Input:
        - legal_cards: list of local legal cards from the runner.
        - state_view: local player view dictionary containing a full `state`.

        Output:
        - A local `Card` object that must be a member of `legal_cards`.
        """
        state = state_view.get("state")
        if state is None:
            raise ValueError("GoPlayerAdapter.play_card requires state_view['state']")
        go_state = to_go_state(state)
        raw_card = self._go_player.choose_card(go_state)
        for local_card in legal_cards:
            if (
                local_card.rank.value == raw_card.rank.value
                and local_card.suit.name == raw_card.suit.name
            ):
                return local_card
        raise ValueError(f"Wrapped collaborator player returned illegal card: {raw_card!r}")


class OurHandStrengthMCTSPlayer(AIPlayer):
    """Local player that uses hand-strength bidding and truncated MCTS play."""

    def __init__(self, config: TruncatedMCTSConfig | None = None) -> None:
        """Create the local bid/play hybrid player.

        Input:
        - config: optional `TruncatedMCTSConfig` for the play strategy.

        Output:
        - A local `AIPlayer` that bids with `strategy.hand_strength` and plays
          with `strategy.truncated_mcts_strategy`.
        """
        self.strategy = TruncatedMCTSStrategy(config)
        self.position = -1
        self.hand: list[LocalCard] = []

    def start_game(self, position: int, hand: list[LocalCard], num_players: int) -> None:
        """Store the local seat and starting hand.

        Input:
        - position: local seat index.
        - hand: list of local cards at deal time.
        - num_players: number of players in the hand.

        Output:
        - None.
        """
        self.position = position
        self.hand = list(hand)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """Use the hand-strength heuristic to choose a legal bid.

        Input:
        - legal_bids: the local legal bid list.
        - state_view: local player view dictionary containing a full `state`.

        Output:
        - One of `legal_bids`.
        """
        state = state_view.get("state")
        if state is None:
            raise ValueError("OurHandStrengthMCTSPlayer.place_bid requires state_view['state']")
        if self.position < 0:
            raise ValueError("OurHandStrengthMCTSPlayer.start_game was not called")

        hand = list(state.hands[self.position])
        # `_hand_strength` expects cards as (suit_char, rank_char) tuples.
        hand_for_strength = [(card.suit.short, card.rank.short) for card in hand]
        bid, _raw_strength = _hand_strength(hand_for_strength)
        if bid == 0:
            if "nil" in legal_bids:
                return "nil"
            if "blind_nil" in legal_bids:
                return "blind_nil"
        return normalize_bid_for_legal_options(bid, legal_bids)

    def play_card(self, legal_cards: list[LocalCard], state_view: dict) -> LocalCard:
        """Choose a card with truncated MCTS and return a legal local card.

        Input:
        - legal_cards: the local legal card list.
        - state_view: local player view dictionary containing a full `state`.

        Output:
        - A local `Card` object that must be a member of `legal_cards`.
        """
        state = state_view.get("state")
        if state is None:
            raise ValueError("OurHandStrengthMCTSPlayer.play_card requires state_view['state']")
        action = self.strategy.choose_action(state)
        if action is None:
            if not legal_cards:
                raise ValueError("No legal cards available")
            return legal_cards[0]
        for local_card in legal_cards:
            if (
                local_card.rank.value == action.rank.value
                and local_card.suit.value == action.suit.value
            ):
                return local_card
        raise ValueError(f"Truncated MCTS returned illegal card: {action!r}")
