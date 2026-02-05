import unittest

from rlcard.games.spades.game import SpadesGame as Game


def _snapshot_game(game):
    return {
        'phase': game.phase,
        'game_pointer': game.game_pointer,
        'dealer_idx': game.dealer_idx,
        'start_player_idx': game.start_player_idx,
        'blind_option_passed': list(game.blind_option_passed),
        'bids_collected': game.bids_collected,
        'tricks_played': game.tricks_played,
        'spades_broken': game.spades_broken,
        'hands': [[card.get_index() for card in p.hand] for p in game.players],
        'player_state': [
            {
                'bid': p.bid,
                'is_nil': p.is_nil,
                'is_blind_nil': p.is_blind_nil,
                'tricks': p.tricks,
            }
            for p in game.players
        ],
        'current_trick': [(pid, card.get_index()) for pid, card in game.round.current_trick],
        'played_cards': [card.get_index() for card in game.round.played_cards],
    }


class TestSpadesGame(unittest.TestCase):
    def test_step_back(self):
        game = Game(allow_step_back=True)
        state, player_id = game.init_game()
        snapshot = _snapshot_game(game)

        action = state['actions'][0]
        game.step(action)
        game.step_back()

        self.assertEqual(game.game_pointer, player_id)
        self.assertEqual(game.history, [])
        self.assertEqual(game.phase, snapshot['phase'])
        self.assertEqual(game.game_pointer, snapshot['game_pointer'])
        self.assertEqual(game.dealer_idx, snapshot['dealer_idx'])
        self.assertEqual(game.start_player_idx, snapshot['start_player_idx'])
        self.assertEqual(game.blind_option_passed, snapshot['blind_option_passed'])
        self.assertEqual(game.bids_collected, snapshot['bids_collected'])
        self.assertEqual(game.tricks_played, snapshot['tricks_played'])
        self.assertEqual(game.spades_broken, snapshot['spades_broken'])
        self.assertEqual(
            [[card.get_index() for card in p.hand] for p in game.players],
            snapshot['hands'],
        )
        self.assertEqual(
            [
                {
                    'bid': p.bid,
                    'is_nil': p.is_nil,
                    'is_blind_nil': p.is_blind_nil,
                    'tricks': p.tricks,
                }
                for p in game.players
            ],
            snapshot['player_state'],
        )
        self.assertEqual(
            [(pid, card.get_index()) for pid, card in game.round.current_trick],
            snapshot['current_trick'],
        )
        self.assertEqual(
            [card.get_index() for card in game.round.played_cards],
            snapshot['played_cards'],
        )

        self.assertEqual(game.step_back(), False)


if __name__ == '__main__':
    unittest.main()
