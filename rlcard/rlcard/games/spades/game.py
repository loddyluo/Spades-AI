import numpy as np
from rlcard.games.spades.dealer import SpadesDealer
from rlcard.games.spades.player import SpadesPlayer
from rlcard.games.spades.judger import SpadesJudger

class SpadesGame:
    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 4
        self.dealer_idx = -1 # Will be incremented to 0 in first init_game if we want random first or 0.
        # Rules say: First random, then rotate. 
        # For RL simplicity, maybe just starts at 0? 
        # The user doc said: "First game random dealer, then rotate". 
        # I will implement random dealer in first call.

    def configure(self, game_config):
        self.num_players = game_config.get('game_num_players', 4)

    def init_game(self):
        # Initialize Dealer and Players
        self.dealer = SpadesDealer(self.np_random)
        self.players = [SpadesPlayer(i, self.np_random) for i in range(self.num_players)]
        self.judger = SpadesJudger(self.np_random)

        # Dealer rotation
        if self.dealer_idx == -1:
            self.dealer_idx = self.np_random.randint(0, self.num_players)
        else:
            self.dealer_idx = (self.dealer_idx + 1) % self.num_players

        # Deal cards
        self.dealer.shuffle()
        self.dealer.deal_cards(self.players)
        # Note: In Blind phase cards are hidden from OBS, but players have them in object.
        
        # Initialize Game State
        self.history = [] # For step_back
        
        # Phases: 0=Blind/Bid, 1=Play
        self.phase = 0 
        
        self.start_player_idx = (self.dealer_idx + 1) % self.num_players
        self.game_pointer = self.start_player_idx
        
        # Tracking bidding status
        # For each player, we need to know if they passed the blind option
        self.blind_option_passed = [False] * self.num_players
        self.bids_collected = 0
        
        # Tracking play status
        self.tricks_played = 0
        self.round = SpadesRound(self.start_player_idx, self.num_players, self.np_random)
        self.spades_broken = False
        
        # Return state
        return self.get_state(self.game_pointer), self.game_pointer

    def step(self, action):
        if self.allow_step_back:
            # Not fully implementing deepcopy for step_back in this initial version for brevity/complexity.
            # But standard RLCard games usually do.
            pass

        current_player = self.players[self.game_pointer]
        
        # Phase 0: Bidding (Blind or Normal)
        if self.phase == 0:
            if action == 'pass': # Pass Blind Nil
                self.blind_option_passed[self.game_pointer] = True
                # Game pointer does NOT move. Player must now bid.
            elif action == 'blind_nil':
                current_player.bid = 0
                current_player.is_blind_nil = True
                self.bids_collected += 1
                self.game_pointer = (self.game_pointer + 1) % self.num_players
            elif action == 'nil' or action.startswith('bid_'):
                # Normal bid
                if action == 'nil':
                    val = 0
                    current_player.is_nil = True
                else:
                    val = int(action.split('_')[1])
                
                current_player.bid = val
                self.bids_collected += 1
                self.game_pointer = (self.game_pointer + 1) % self.num_players
            
            # Check if bidding ended
            if self.bids_collected == self.num_players:
                self.phase = 1
                # Play starts from Dealer Left (which is start_player_idx)
                self.game_pointer = self.start_player_idx
                self.round.start_new_trick(self.game_pointer)
                
        # Phase 1: Playing
        elif self.phase == 1:
            # Action is a card string presumably, or Card object? 
            # In Env we decode to string usually. Let's assume action is rank+suit str e.g. "SA" or "2H".
            # Or "AS", "H2"? base.py Card str is Rank+Suit (e.g. "AS").
            
            # Find card in hand
            card_to_play = None
            for card in current_player.hand:
                if card.get_index() == action:
                    card_to_play = card
                    break
            
            # Remove card from hand
            current_player.hand.remove(card_to_play)
            
            # Check Spades Broken
            if card_to_play.suit == 'S':
                 # If not lead, it breaks spades
                 if len(self.round.current_trick) > 0: # Not leading
                     lead_suit = self.round.current_trick[0][1].suit
                     if lead_suit != 'S':
                         self.spades_broken = True
                 else: # Leading spade
                     # Leading spade implies broken or forced.
                     self.spades_broken = True

            # Add to round
            self.round.play_card(self.game_pointer, card_to_play)
            
            # If trick finished
            if len(self.round.current_trick) == self.num_players:
                winner_id = self.judger.judge_trick(self)
                self.players[winner_id].tricks += 1
                self.tricks_played += 1
                
                if self.tricks_played == 13:
                    # Game Over
                    return self.get_state(self.game_pointer), self.game_pointer # State will show game over
                else:
                    # Next trick
                    self.game_pointer = winner_id
                    self.round.start_new_trick(self.game_pointer)
            else:
                self.game_pointer = (self.game_pointer + 1) % self.num_players

        return self.get_state(self.game_pointer), self.game_pointer

    def get_state(self, player_id):
        ''' Return player's state
        '''
        state = {}
        player = self.players[player_id]
        
        # Populate basic info
        state['hand'] = [c.get_index() for c in player.hand]
        state['others_hand'] = [] # Usually obscured
        state['current_player'] = player_id
        state['dealer'] = self.dealer_idx
        state['phase'] = self.phase
        
        # Bidding info
        state['bids'] = [p.bid for p in self.players] # List of None or Int
        state['blind_passed'] = self.blind_option_passed[:]
        
        # Play info
        state['trick'] = [(p, c.get_index()) for p, c in self.round.current_trick]
        state['spades_broken'] = self.spades_broken
        state['tricks_won'] = [p.tricks for p in self.players]
        state['played_cards'] = [c.get_index() for c in self.round.played_cards]
        
        # Actions
        state['actions'] = self.get_legal_actions(player_id)
        
        return state

    def get_legal_actions(self, player_id):
        actions = []
        player = self.players[player_id]
        
        if self.phase == 0:
            # Bidding
            if not self.blind_option_passed[player_id] and not player.bid is not None: 
                # Has not passed blind option yet, and hasn't bid (bid is None)
                # Available: blind_nil, pass
                actions = ['blind_nil', 'pass']
            else:
                # Must bid normal
                actions = ['nil'] + [f'bid_{i}' for i in range(1, 14)]
        else:
            # Playing
            # Standard Spades Rules
            current_hand = player.hand
            if len(self.round.current_trick) == 0:
                # Leading
                # Can lead Spade ONLY if spades_broken OR only has spades
                has_non_spade = any(c.suit != 'S' for c in current_hand)
                for card in current_hand:
                    if card.suit == 'S':
                        if self.spades_broken or not has_non_spade:
                            actions.append(card.get_index())
                    else:
                        actions.append(card.get_index())
            else:
                # Following
                lead_card = self.round.current_trick[0][1]
                lead_suit = lead_card.suit
                has_lead_suit = any(c.suit == lead_suit for c in current_hand)
                
                for card in current_hand:
                    if has_lead_suit:
                        if card.suit == lead_suit:
                            actions.append(card.get_index())
                    else:
                        actions.append(card.get_index())
        return actions

    def is_over(self):
        return self.tricks_played >= 13

    def get_num_players(self):
        return self.num_players

    def get_num_actions(self):
        # 52 cards + Pass + Blind Nil + Nil + 13 Bids = 52 + 1 + 1 + 1 + 13 = 68
        return 68

    def get_player_id(self):
        return self.game_pointer

class SpadesRound:
    def __init__(self, start_player, num_players, np_random):
        self.start_player = start_player
        self.num_players = num_players
        self.np_random = np_random
        self.current_trick = [] # List of (player_id, Card)
        self.played_cards = []
        
    def start_new_trick(self, leader):
        self.current_trick = []
        self.start_player = leader
        
    def play_card(self, player_id, card):
        self.current_trick.append((player_id, card))
        self.played_cards.append(card)
