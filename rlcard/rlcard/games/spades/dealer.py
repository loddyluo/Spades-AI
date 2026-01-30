from rlcard.utils import init_standard_deck

class SpadesDealer:
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_standard_deck()
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        ''' Deal cards to players
        '''
        hand_size = 13
        for i, player in enumerate(players):
            player.hand = self.deck[i*hand_size : (i+1)*hand_size]
            player.hand.sort(key=lambda card: (card.suit, card.rank)) # Optional: sort for easier viewing usually?
            # Note: Card object sorting depends on implementation. 
            # In base.py Card does not have cmp defined for Spades logic specifically. 
            # We will handle sorting in extract_state if needed or implement a sort key.
