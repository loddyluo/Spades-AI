from spades_ai.game.card import Card
from spades_ai.game.state import GameState, Bid
from spades_ai.game.legal_moves import get_legal_moves
from spades_ai.encoding.encoder import ObservationEncoder
from spades_ai.search.go_mcts import GOMCTSSearch, GOMCTSConfig


class GOMCTSPlayer:
    def __init__(self, model, config=None, device=None):
        cfg = config if config is not None else GOMCTSConfig()
        self._search = GOMCTSSearch(model, cfg, device)
        self._encoder = ObservationEncoder()

    def choose_bid(self, state):
        from spades_ai.players.rule_based.bidding import rule_based_bid
        p = state.current_player
        return rule_based_bid(state.hands[p], state.bids, p)

    def choose_card(self, state):
        p = state.current_player
        legal = get_legal_moves(
            state.hands[p],
            state.led_suit,
            state.spades_broken,
            len(state.current_trick_cards) == 0,
        )
        if len(legal) == 1:
            return next(iter(legal))
        tokens = self._encoder.encode(state, p)
        card, _ = self._search.run(tokens, legal, p)
        return card
