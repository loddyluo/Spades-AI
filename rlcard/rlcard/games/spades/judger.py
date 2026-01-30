class SpadesJudger:
    def __init__(self, np_random):
        self.np_random = np_random

    def judge_game(self, players):
        ''' Judge the game and return payoffs
        
        Args:
            players (list): list of SpadesPlayer
            
        Returns:
            payoffs (list): payoff for each player
        '''
        # Calculate team scores
        # Team 0: Player 0 and 2
        # Team 1: Player 1 and 3
        
        team_scores = {0: 0, 1: 0}
        
        # Calculate for Team 0
        score0 = self.calculate_team_score(players[0], players[2])
        team_scores[0] = score0
        
        # Calculate for Team 1
        score1 = self.calculate_team_score(players[1], players[3])
        team_scores[1] = score1
        
        # Payoffs are the team scores
        return [team_scores[0], team_scores[1], team_scores[0], team_scores[1]]

    def calculate_team_score(self, p1, p2):
        score = 0
        team_tricks = p1.tricks + p2.tricks
        team_bid = 0
        
        # Process p1
        if p1.is_blind_nil:
            if p1.tricks == 0:
                score += 100
            else:
                score -= 100
        elif p1.is_nil:
            if p1.tricks == 0:
                score += 50
            else:
                score -= 50
        else:
            team_bid += p1.bid
            
        # Process p2
        if p2.is_blind_nil:
            if p2.tricks == 0:
                score += 100
            else:
                score -= 100
        elif p2.is_nil:
            if p2.tricks == 0:
                score += 50
            else:
                score -= 50
        else:
            team_bid += p2.bid
            
        # Process Normal Contract
        # If both are nil/blind nil, team_bid is 0. 
        # Standard rules usually say if team bid is 0, no extra points/penalty for bags?
        # But here rules say "Nil 失败的墩数... 按正常规则计算分数". 
        # If team_bid is 0 (double nil), and they get tricks, those are technically overtricks for a bid of 0?
        # Or maybe Team Bid of 0 constraint is not possible in standard play usually?
        # However, following the rule: "If Team Tricks < Team Bid: - (Team Bid * 10)"
        # If Team Bid is 0, this logic implies 0 score for the contract part.
        
        if team_bid > 0:
            if team_tricks < team_bid:
                score -= (team_bid * 10)
            else:
                overtricks = team_tricks - team_bid
                # Rule: "Contract * 10 + Overtrick * (-9)"
                score += (team_bid * 10) - (overtricks * 9)
        
        return score

    def judge_trick(self, game):
        ''' Judge the winner of a trick
        
        Args:
            game (SpadesGame): current game instance
            
        Returns:
            winner_id (int): index of the winner
        '''
        # game.round.current_trick contains (player_id, card) tuples
        trick_cards = game.round.current_trick
        if not trick_cards:
            return None
            
        # Find lead suit
        lead_card = trick_cards[0][1]
        lead_suit = lead_card.suit
        
        highest_trump = None
        highest_lead = None
        
        winner_idx = 0
        
        for i, (pid, card) in enumerate(trick_cards):
            is_trump = (card.suit == 'S')
            is_lead = (card.suit == lead_suit)
            
            # Rank comparison helper
            # 2-9, T, J, Q, K, A. 
            # We can use card.rank directly if we map them or use base.py card convention?
            # base.py Card does not maintain numeric value.
            # We need a rank value mapper.
            
            val = self.get_rank_value(card.rank)
            
            if is_trump:
                if highest_trump is None or val > highest_trump[1]:
                    highest_trump = (i, val)
            elif is_lead:
                if highest_trump is None: # Only care if no trump seen yet or compared to lead
                    if highest_lead is None or val > highest_lead[1]:
                        highest_lead = (i, val)
        
        # Determine winner
        if highest_trump:
            winner_idx = highest_trump[0]
        elif highest_lead:
            winner_idx = highest_lead[0]
        else:
            # Should not happen as someone played the lead card
            winner_idx = 0 
            
        return trick_cards[winner_idx][0]

    @staticmethod
    def get_rank_value(rank):
        # 2 < 3 ... < 9 < T < J < Q < K < A
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        if rank in ranks:
            return ranks.index(rank)
        return -1
