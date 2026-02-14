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
        ''' Extract observable state
        '''
        obs_vec = np.zeros(200, dtype=np.int8) # Arbitrary large enough safe size
        
        # 1. Hand (0-51)
        # If Phase 0 (Bidding) and Blind Option not passed, MASK hand.
        player_id = state['current_player']
        if self.enable_blind_nil and state['phase'] == 0 and not state['blind_passed'][player_id]:
            # Mask hand (keep zeros)
            pass 
        else:
            current_hand = state['hand']
            for card_str in current_hand:
                if card_str in self.card2id:
                    obs_vec[self.card2id[card_str]] = 1
        
        # 2. Bids (52-67 reserved for bid info? No, can use values directly or one-hot)
        # Bids: 4 players. Values 0-13, or -1 (None). 
        # Using normalized values or one-hot? 
        # Simple: 4 slots for bids. -1 if not bid yet.
        # Let's use 52 + 4 indices.
        offset = 52
        for i, bid in enumerate(state['bids']):
            if bid is not None:
                # Store as value (normalized? or raw?)
                # Raw is fine for some models, but one-hot preferred? 
                # Let's just put value for now as simple vector.
                obs_vec[offset + i] = bid
            else:
                obs_vec[offset + i] = -1 # Or 14?
        
        # 3. Tricks won (4 slots)
        offset += 4
        for i, t in enumerate(state['tricks_won']):
            obs_vec[offset + i] = t
            
        # 4. Spades broken (1 slot)
        offset += 4
        obs_vec[offset] = 1 if state['spades_broken'] else 0
        
        # 5. Cards played in current trick (52 one-hot)
        offset += 1
        current_trick = state['trick']
        for p, c_str in current_trick:
             if c_str in self.card2id:
                 obs_vec[offset + self.card2id[c_str]] = 1
                 
        # Construct dictionary
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
            value = team0_score - self.reward_beta * team1_score
            return np.array([value, value, value, value])
        return np.array(raw_payoffs)
