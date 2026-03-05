import numpy as np
from collections import OrderedDict
from rlcard.envs import Env
from rlcard.games.spades.game import SpadesGame as Game
from rlcard.games.base import Card

class SpadesEnv(Env):
    def __init__(self, config):
        self.name = 'spades'
        self.game = Game(enable_blind_nil=config.get('game_enable_blind_nil', True))
        super().__init__(config)
        self.reward_beta = config.get('reward_beta', 1.0)
        self.enable_blind_nil = config.get('game_enable_blind_nil', True)
        
        # Calculate action map
        self.actions = []
        # 1. Cards 0-51
        self.card2id = {}
        self.id2card = {}
        idx = 0
        suites = ['S', 'H', 'D', 'C']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        for suit in suites:
            for rank in ranks:
                 # Card get_index returns Suit+Rank e.g. "S2" or "SA". 
                 # Wait, base.py Card.__str__ is Rank+Suit ("AS"), get_index is Suit+Rank ("SA").
                 # Let's verify base.py output from previous read_file.
                 # "return self.suit+self.rank"
                 card_str = suit + rank 
                 self.actions.append(card_str)
                 self.card2id[card_str] = idx
                 idx += 1
        
        # 2. Special Actions
        self.actions.append('pass') # 52
        self.actions.append('blind_nil') # 53
        self.actions.append('nil') # 54
        for i in range(1, 14):
            self.actions.append(f'bid_{i}') # 55-67
            
        self.state_shape = [[200] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        ''' Extract observable state into a 200-dim int8 vector.

        Layout (200 dimensions total):
          [  0- 51] 52  Hand cards one-hot
          [ 52- 55]  4  Bid values per player (-1 = not yet bid, 0-13)
          [ 56- 59]  4  Tricks won per player
          [ 60    ]  1  Spades broken flag
          [ 61-112] 52  Current trick cards one-hot
          [113-164] 52  Played-cards history one-hot (NEW)
          [165-168]  4  is_nil flag per player (NEW)
          [169-172]  4  is_blind_nil flag per player (NEW)
          [173-176]  4  Current player id one-hot (NEW)
          [177-180]  4  Trick card owners — position i=1 if player i
                        has a card in the current trick (NEW)
          [181-184]  4  Hand size per player (NEW)
          [185-199] 15  Reserved (zeros)
        '''
        obs_vec = np.zeros(200, dtype=np.int8)

        player_id = state['current_player']

        # 1. Hand (0-51) — masked during blind-nil decision
        if self.enable_blind_nil and state['phase'] == 0 and not state['blind_passed'][player_id]:
            pass  # keep zeros
        else:
            for card_str in state['hand']:
                if card_str in self.card2id:
                    obs_vec[self.card2id[card_str]] = 1

        # 2. Bids (52-55)
        offset = 52
        for i, bid in enumerate(state['bids']):
            obs_vec[offset + i] = bid if bid is not None else -1

        # 3. Tricks won (56-59)
        offset = 56
        for i, t in enumerate(state['tricks_won']):
            obs_vec[offset + i] = t

        # 4. Spades broken (60)
        obs_vec[60] = 1 if state['spades_broken'] else 0

        # 5. Current trick cards one-hot (61-112)
        offset = 61
        for _pid, c_str in state['trick']:
            if c_str in self.card2id:
                obs_vec[offset + self.card2id[c_str]] = 1

        # 6. Played-cards history one-hot (113-164)  — NEW
        offset = 113
        for card_str in state['played_cards']:
            if card_str in self.card2id:
                obs_vec[offset + self.card2id[card_str]] = 1

        # 7. Nil flags per player (165-168)  — NEW
        offset = 165
        for i, v in enumerate(state.get('is_nil', [False] * 4)):
            obs_vec[offset + i] = 1 if v else 0

        # 8. Blind-nil flags per player (169-172)  — NEW
        offset = 169
        for i, v in enumerate(state.get('is_blind_nil', [False] * 4)):
            obs_vec[offset + i] = 1 if v else 0

        # 9. Current player id one-hot (173-176)  — NEW
        obs_vec[173 + player_id] = 1

        # 10. Trick card owners (177-180)  — NEW
        for pid, _c_str in state['trick']:
            obs_vec[177 + pid] = 1

        # 11. Hand sizes per player (181-184)  — NEW
        offset = 181
        for i, sz in enumerate(state.get('hand_sizes', [13] * 4)):
            obs_vec[offset + i] = sz

        # Construct legal-action dict
        legal_actions = OrderedDict()
        for action_str in state['actions']:
            if action_str in self.actions:
                idx = self.actions.index(action_str)
                legal_actions[idx] = None

        extracted_state = {
            'obs': obs_vec,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': state['actions'],
            'action_record': self.action_recorder
        }
        return extracted_state

    def _decode_action(self, action_id):
        return self.actions[action_id]

    def _get_legal_actions(self):
        # Env logic calls game directly usually in step, 
        # but rlcard logic sometimes calls this internally?
        # game.step returns state with legal actions already.
        # This function is used if we need to query form Env without stepping?
        # We can implement using game.get_legal_actions(game.game_pointer)
        return self.game.get_legal_actions(self.game.game_pointer)

    def get_payoffs(self):
        raw_payoffs = self.game.judger.judge_game(self.game.players)
        if len(raw_payoffs) == 4:
            team0_score = raw_payoffs[0]
            team1_score = raw_payoffs[1]
            value_team0 = team0_score - self.reward_beta * team1_score
            value_team1 = team1_score - self.reward_beta * team0_score
            return np.array([value_team0, value_team1, value_team0, value_team1])
        return np.array(raw_payoffs)
