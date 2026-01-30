class SpadesPlayer:
    def __init__(self, player_id, np_random):
        ''' Initialize a Spades player class

        Args:
            player_id (int): id for the player
            np_random (numpy.random.RandomState): random state
        '''
        self.player_id = player_id
        self.np_random = np_random
        self.hand = []
        
        # Bidding state
        self.bid = None # 0-13 or 'Blind Nil' (marked specially, maybe -1 or separate flag)
        self.is_blind_nil = False
        self.is_nil = False
        
        # Play state
        self.tricks = 0
        self.won_tricks = [] # Store won cards if needed? Probably just count is enough for scoring.

    def get_player_id(self):
        ''' Return player's id
        '''
        return self.player_id
